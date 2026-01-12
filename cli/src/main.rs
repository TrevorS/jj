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
use jj_lib::object_id::ObjectId;
use jj_lib::repo::StoreFactories;
use jj_lib::signing::Signer;
use jj_lib::workspace::Workspace;
use jj_lib::workspace::WorkspaceInitError;
use jjhub_backend::JjhubBackend;
use jjhub_backend::JjhubClient;
use jjhub_backend::JjhubCommitId;
use jjhub_backend::JjhubConfig;

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
    /// Fetch changes and bookmarks from the server
    Fetch(JjhubFetchArgs),
    /// Push bookmarks to the server
    Push(JjhubPushArgs),
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

#[derive(clap::Args, Clone, Debug)]
struct JjhubFetchArgs {
    /// Specific bookmarks to fetch (fetches all if not specified)
    #[arg(long, short = 'B')]
    bookmark: Vec<String>,
}

#[derive(clap::Args, Clone, Debug)]
struct JjhubPushArgs {
    /// Specific bookmarks to push (pushes all if not specified)
    #[arg(long, short = 'B')]
    bookmark: Vec<String>,
    /// Create the bookmark on the remote if it doesn't exist
    #[arg(long)]
    create: bool,
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
            JjhubCommand::Fetch(args) => run_jjhub_fetch(ui, command_helper, args),
            JjhubCommand::Push(args) => run_jjhub_push(ui, command_helper, args),
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

fn run_jjhub_fetch(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    args: JjhubFetchArgs,
) -> Result<(), CommandError> {
    let workspace_command = command_helper.workspace_helper(ui)?;
    let store_path = workspace_command.repo_path().join("store");

    // Load jjhub config to get server/owner/repo
    let config = JjhubConfig::load(&store_path).map_err(|e| {
        user_error(format!(
            "Not a jjhub repository (failed to load config): {}",
            e
        ))
    })?;

    let client = JjhubClient::new(&config.base_url);

    // List remote bookmarks
    let remote_bookmarks = client.list_bookmarks(&config.owner, &config.repo).map_err(|e| {
        user_error(format!("Failed to list remote bookmarks: {}", e))
    })?;

    // Filter bookmarks if specific ones requested
    let bookmarks_to_fetch: Vec<_> = if args.bookmark.is_empty() {
        remote_bookmarks
    } else {
        remote_bookmarks
            .into_iter()
            .filter(|b| args.bookmark.contains(&b.name))
            .collect()
    };

    if bookmarks_to_fetch.is_empty() {
        writeln!(ui.status(), "No bookmarks to fetch")?;
        return Ok(());
    }

    writeln!(
        ui.status(),
        "Fetching {} bookmark(s) from {}/{}",
        bookmarks_to_fetch.len(),
        config.owner,
        config.repo
    )?;

    for bookmark_summary in &bookmarks_to_fetch {
        // Get full bookmark details
        match client.get_bookmark(&config.owner, &config.repo, &bookmark_summary.name) {
            Ok(Some(bookmark)) => {
                let status = if bookmark.is_conflicted() {
                    " (conflicted)"
                } else {
                    ""
                };
                writeln!(
                    ui.status(),
                    "  {} -> {} target(s){}",
                    bookmark.name,
                    bookmark.targets().len(),
                    status
                )?;
            }
            Ok(None) => {
                writeln!(ui.status(), "  {} (not found)", bookmark_summary.name)?;
            }
            Err(e) => {
                writeln!(
                    ui.warning_default(),
                    "{} (error: {})",
                    bookmark_summary.name,
                    e
                )?;
            }
        }
    }

    writeln!(ui.status(), "Fetch complete")?;
    Ok(())
}

fn run_jjhub_push(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    args: JjhubPushArgs,
) -> Result<(), CommandError> {
    let workspace_command = command_helper.workspace_helper(ui)?;
    let store_path = workspace_command.repo_path().join("store");

    // Load jjhub config
    let config = JjhubConfig::load(&store_path).map_err(|e| {
        user_error(format!(
            "Not a jjhub repository (failed to load config): {}",
            e
        ))
    })?;

    let client = JjhubClient::new(&config.base_url);
    let repo = workspace_command.repo();

    // Get local bookmarks from the view
    let view = repo.view();
    let local_bookmarks: Vec<_> = view
        .bookmarks()
        .filter_map(|(name, target)| {
            // Only include bookmarks that have local targets
            if target.local_target.is_present() {
                Some(name.to_owned())
            } else {
                None
            }
        })
        .collect();

    // Filter bookmarks if specific ones requested
    let bookmarks_to_push: Vec<_> = if args.bookmark.is_empty() {
        local_bookmarks
    } else {
        local_bookmarks
            .into_iter()
            .filter(|name| args.bookmark.iter().any(|b| b == name.as_str()))
            .collect()
    };

    if bookmarks_to_push.is_empty() {
        writeln!(ui.status(), "No bookmarks to push")?;
        return Ok(());
    }

    writeln!(
        ui.status(),
        "Pushing {} bookmark(s) to {}/{}",
        bookmarks_to_push.len(),
        config.owner,
        config.repo
    )?;

    for bookmark_name in &bookmarks_to_push {
        let name_str = bookmark_name.as_str();
        let local_target = view.get_local_bookmark(bookmark_name);

        // Get commit IDs from the local target
        let commit_ids: Vec<_> = local_target
            .added_ids()
            .map(|id| JjhubCommitId::from_hex(&id.hex()).expect("valid commit id"))
            .collect();

        if commit_ids.is_empty() {
            writeln!(ui.warning_default(), "{} has no targets, skipping", name_str)?;
            continue;
        }

        // Check if bookmark exists on remote
        let remote_bookmark = client
            .get_bookmark(&config.owner, &config.repo, name_str)
            .map_err(|e| user_error(format!("Failed to check remote bookmark: {}", e)))?;

        let expected_version = remote_bookmark.as_ref().map(|b| b.version);

        if remote_bookmark.is_none() && !args.create {
            writeln!(
                ui.warning_default(),
                "{} does not exist on remote (use --create to create it)",
                name_str
            )?;
            continue;
        }

        // Push the bookmark
        match client.set_bookmark(
            &config.owner,
            &config.repo,
            name_str,
            &commit_ids,
            expected_version,
        ) {
            Ok(bookmark) => {
                let action = if remote_bookmark.is_some() {
                    "updated"
                } else {
                    "created"
                };
                writeln!(
                    ui.status(),
                    "  {} {} (version {})",
                    name_str,
                    action,
                    bookmark.version
                )?;
            }
            Err(e) => {
                writeln!(ui.warning_default(), "{} failed: {}", name_str, e)?;
            }
        }
    }

    writeln!(ui.status(), "Push complete")?;
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
