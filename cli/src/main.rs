// Copyright 2020 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::path::PathBuf;

use jj_cli::cli_util::CliRunner;
use jj_cli::cli_util::CommandHelper;
use jj_cli::command_error::CommandError;
use jj_cli::command_error::user_error;
use jj_cli::ui::Ui;
use jj_lib::repo::StoreFactories;
use jj_lib::signing::Signer;
use jj_lib::workspace::Workspace;
use jj_lib::workspace::WorkspaceInitError;
use jjhub_backend::factory::JjhubConfig;
use jjhub_backend::JjhubBackend;

/// Create store factories with jjhub backend registered.
fn create_store_factories() -> StoreFactories {
    let mut store_factories = StoreFactories::empty();
    store_factories.add_backend(
        JjhubBackend::NAME,
        Box::new(|_settings, store_path| {
            jjhub_backend::factory::load(store_path)
                .map_err(|e| jj_lib::backend::BackendLoadError(Box::new(e)))
        }),
    );
    store_factories
}

/// Root level custom commands (adds `jjhub` subcommand to jj)
#[derive(clap::Subcommand, Clone, Debug)]
enum CustomCommand {
    /// Jjhub: native code hosting for Jujutsu
    #[command(subcommand)]
    Jjhub(JjhubCommand),
}

/// Jjhub subcommands (nested under `jj jjhub`)
#[derive(clap::Subcommand, Clone, Debug)]
enum JjhubCommand {
    /// Initialize a jjhub-backed repository
    Init(JjhubInitArgs),
    /// Clone a repository from a jjhub server
    Clone(JjhubCloneArgs),
}

#[derive(clap::Args, Clone, Debug)]
struct JjhubInitArgs {
    /// jjhub server URL (e.g., https://jjhub.example.com)
    #[arg(long)]
    server: String,
    /// Repository owner
    #[arg(long)]
    owner: String,
    /// Repository name
    #[arg(long)]
    repo: String,
}

#[derive(clap::Args, Clone, Debug)]
struct JjhubCloneArgs {
    /// jjhub repository URL (e.g., https://jjhub.example.com/owner/repo)
    url: String,
    /// Destination directory (defaults to repo name)
    destination: Option<PathBuf>,
}

fn run_custom_command(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    command: CustomCommand,
) -> Result<(), CommandError> {
    match command {
        CustomCommand::Jjhub(jjhub_cmd) => match jjhub_cmd {
            JjhubCommand::Init(args) => run_jjhub_init(ui, command_helper, args),
            JjhubCommand::Clone(args) => run_jjhub_clone(ui, command_helper, args),
        },
    }
}

fn run_jjhub_init(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    args: JjhubInitArgs,
) -> Result<(), CommandError> {
    let wc_path = command_helper.cwd();
    let settings = command_helper.settings_for_new_workspace(wc_path)?;

    let config = JjhubConfig::new(&args.server, &args.owner, &args.repo);

    // Initialize workspace with jjhub backend
    Workspace::init_with_backend(
        &settings,
        wc_path,
        &|_settings, store_path| {
            jjhub_backend::factory::init(store_path, &config)
                .map_err(|e| jj_lib::backend::BackendInitError(Box::new(e)))?;
            jjhub_backend::factory::load(store_path)
                .map_err(|e| jj_lib::backend::BackendInitError(Box::new(e)))
        },
        Signer::from_settings(&settings).map_err(WorkspaceInitError::SignInit)?,
    )?;

    writeln!(
        ui.status(),
        "Initialized jjhub repository for {}/{}",
        args.owner,
        args.repo
    )?;
    writeln!(ui.status(), "Server: {}", args.server)?;
    Ok(())
}

fn run_jjhub_clone(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    args: JjhubCloneArgs,
) -> Result<(), CommandError> {
    // Parse URL: https://host/owner/repo or jjhub://host/owner/repo
    let (base_url, owner, repo) = parse_jjhub_url(&args.url)?;

    let dest = args
        .destination
        .unwrap_or_else(|| PathBuf::from(&repo));

    // Create destination directory
    if dest.exists() {
        return Err(user_error(format!(
            "Destination path '{}' already exists",
            dest.display()
        )));
    }
    std::fs::create_dir_all(&dest).map_err(|e| {
        user_error(format!(
            "Failed to create directory '{}': {}",
            dest.display(),
            e
        ))
    })?;

    let settings = command_helper.settings_for_new_workspace(&dest)?;
    let config = JjhubConfig::new(&base_url, &owner, &repo);

    // Initialize workspace with jjhub backend
    Workspace::init_with_backend(
        &settings,
        &dest,
        &|_settings, store_path| {
            jjhub_backend::factory::init(store_path, &config)
                .map_err(|e| jj_lib::backend::BackendInitError(Box::new(e)))?;
            jjhub_backend::factory::load(store_path)
                .map_err(|e| jj_lib::backend::BackendInitError(Box::new(e)))
        },
        Signer::from_settings(&settings).map_err(WorkspaceInitError::SignInit)?,
    )?;

    writeln!(ui.status(), "Cloned {}/{} to {:?}", owner, repo, dest)?;
    Ok(())
}

/// Parse a jjhub URL into (base_url, owner, repo).
fn parse_jjhub_url(url: &str) -> Result<(String, String, String), CommandError> {
    if let Some(rest) = url.strip_prefix("jjhub://") {
        // jjhub://host/owner/repo
        let parts: Vec<&str> = rest.splitn(3, '/').collect();
        if parts.len() < 3 {
            return Err(user_error(
                "Invalid jjhub URL. Expected: jjhub://host/owner/repo",
            ));
        }
        Ok((
            format!("https://{}", parts[0]),
            parts[1].to_string(),
            parts[2].to_string(),
        ))
    } else if let Some(rest) = url.strip_prefix("https://") {
        // https://host/owner/repo
        parse_http_url("https", rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        // http://host/owner/repo
        parse_http_url("http", rest)
    } else {
        Err(user_error(
            "Invalid URL scheme. Expected https://, http://, or jjhub://",
        ))
    }
}

fn parse_http_url(scheme: &str, rest: &str) -> Result<(String, String, String), CommandError> {
    // rest is "host/owner/repo" or "host:port/owner/repo"
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() < 3 {
        return Err(user_error(format!(
            "Invalid URL. Expected: {}://host/owner/repo",
            scheme
        )));
    }
    let host = parts[0];
    let owner = parts[1];
    let repo = parts[2].trim_end_matches('/'); // Remove trailing slash if any

    Ok((
        format!("{}://{}", scheme, host),
        owner.to_string(),
        repo.to_string(),
    ))
}

fn main() -> std::process::ExitCode {
    CliRunner::init()
        .version(env!("JJ_VERSION"))
        .add_store_factories(create_store_factories())
        .add_subcommand(run_custom_command)
        .run()
        .into()
}
