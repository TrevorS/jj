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
use jj_lib::backend::CommitId;
use jj_lib::merge::Merge;
use jj_lib::object_id::ObjectId;
use jj_lib::op_store::{RefTarget, RemoteRef, RemoteRefState};
use jj_lib::ref_name::{RefName, RefNameBuf, RemoteName};
use jj_lib::repo::StoreFactories;
use jj_lib::signing::Signer;
use jj_lib::workspace::Workspace;
use jj_lib::workspace::WorkspaceInitError;
use jjhub_backend::JjhubBackend;
use jjhub_backend::JjhubClient;
use jjhub_backend::JjhubCommitId;
use jjhub_backend::JjhubConfig;

/// The default remote name for jjhub.
const JJHUB_REMOTE_NAME: &str = "jjhub";

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
    /// Login to a jjhub server and store authentication token
    Login(JjhubLoginArgs),
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

#[derive(clap::Args, Clone, Debug)]
struct JjhubLoginArgs {
    /// Username to authenticate as
    #[arg(long, short = 'u')]
    username: Option<String>,
    /// Server URL (uses current repository's server if not specified)
    #[arg(long)]
    server: Option<String>,
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
            JjhubCommand::Login(args) => run_jjhub_login(ui, command_helper, args),
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

    let dest = args.destination.unwrap_or_else(|| PathBuf::from(&repo));

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
    let mut workspace_command = command_helper.workspace_helper(ui)?;
    let store_path = workspace_command.repo_path().join("store");

    // Load jjhub config to get server/owner/repo
    let config = JjhubConfig::load(&store_path).map_err(|e| {
        user_error(format!(
            "Not a jjhub repository (failed to load config): {}",
            e
        ))
    })?;

    // Create client with auth token if available
    let client = if let Some(ref token) = config.token {
        JjhubClient::with_token(&config.base_url, token)
    } else {
        JjhubClient::new(&config.base_url)
    };

    // List remote bookmarks
    let remote_bookmarks = client
        .list_bookmarks(&config.owner, &config.repo)
        .map_err(|e| user_error(format!("Failed to list remote bookmarks: {}", e)))?;

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

    // Start a transaction to update remote tracking refs
    let mut tx = workspace_command.start_transaction();
    let remote_name = RemoteName::new(JJHUB_REMOTE_NAME);
    let mut fetched_count = 0;

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

                // Convert jjhub targets to jj-lib RefTarget
                let ref_target = if bookmark.targets().is_empty() {
                    RefTarget::absent()
                } else if bookmark.targets().len() == 1 {
                    // Single target - resolved
                    let commit_id = CommitId::new(bookmark.targets()[0].as_bytes().to_vec());
                    RefTarget::normal(commit_id)
                } else {
                    // Multiple targets - conflicted (merge all adds)
                    let adds: Vec<Option<CommitId>> = bookmark
                        .targets()
                        .iter()
                        .map(|t| Some(CommitId::new(t.as_bytes().to_vec())))
                        .collect();
                    // Merge needs alternating removes and adds, but for a conflict
                    // we just have adds. Use empty removes between them.
                    let removes = vec![None; adds.len().saturating_sub(1)];
                    RefTarget::from_merge(Merge::from_removes_adds(removes, adds))
                };

                // Create RemoteRef for this bookmark
                let remote_ref = RemoteRef {
                    target: ref_target,
                    state: RemoteRefState::Tracked, // Auto-track fetched bookmarks
                };

                // Update the remote tracking ref
                let ref_name = RefName::new(&bookmark.name);
                let symbol = ref_name.to_remote_symbol(&remote_name);
                tx.repo_mut().set_remote_bookmark(symbol, remote_ref);
                fetched_count += 1;
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

    // Finish the transaction if we updated anything
    if fetched_count > 0 {
        tx.finish(
            ui,
            format!(
                "fetch {} bookmark(s) from jjhub {}/{}",
                fetched_count, config.owner, config.repo
            ),
        )?;
    }

    writeln!(ui.status(), "Fetch complete")?;
    Ok(())
}

fn run_jjhub_push(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    args: JjhubPushArgs,
) -> Result<(), CommandError> {
    let mut workspace_command = command_helper.workspace_helper(ui)?;
    let store_path = workspace_command.repo_path().join("store");

    // Load jjhub config
    let config = JjhubConfig::load(&store_path).map_err(|e| {
        user_error(format!(
            "Not a jjhub repository (failed to load config): {}",
            e
        ))
    })?;

    // Create client with auth token if available
    let client = if let Some(ref token) = config.token {
        JjhubClient::with_token(&config.base_url, token)
    } else {
        JjhubClient::new(&config.base_url)
    };
    let repo = workspace_command.repo().clone();

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

    // Collect successfully pushed bookmarks for updating remote tracking refs
    let mut pushed_bookmarks: Vec<(RefNameBuf, RefTarget)> = Vec::new();

    for bookmark_name in &bookmarks_to_push {
        let name_str = bookmark_name.as_str();
        let local_target = view.get_local_bookmark(bookmark_name);

        // Get commit IDs from the local target
        let commit_ids: Vec<_> = local_target
            .added_ids()
            .map(|id| JjhubCommitId::from_hex(&id.hex()).expect("valid commit id"))
            .collect();

        if commit_ids.is_empty() {
            writeln!(
                ui.warning_default(),
                "{} has no targets, skipping",
                name_str
            )?;
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

                // Record for remote tracking update
                pushed_bookmarks.push((name_str.into(), local_target.clone()));
            }
            Err(e) => {
                writeln!(ui.warning_default(), "{} failed: {}", name_str, e)?;
            }
        }
    }

    // Update remote tracking refs for successfully pushed bookmarks
    if !pushed_bookmarks.is_empty() {
        let mut tx = workspace_command.start_transaction();
        let remote_name = RemoteName::new(JJHUB_REMOTE_NAME);

        for (ref_name, target) in &pushed_bookmarks {
            let remote_ref = RemoteRef {
                target: target.clone(),
                state: RemoteRefState::Tracked,
            };
            let symbol = ref_name.to_remote_symbol(&remote_name);
            tx.repo_mut().set_remote_bookmark(symbol, remote_ref);
        }

        tx.finish(
            ui,
            format!(
                "push {} bookmark(s) to jjhub {}/{}",
                pushed_bookmarks.len(),
                config.owner,
                config.repo
            ),
        )?;
    }

    writeln!(ui.status(), "Push complete")?;
    Ok(())
}

fn run_jjhub_login(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    args: JjhubLoginArgs,
) -> Result<(), CommandError> {
    // Determine server URL - either from args or from current repo config
    let (server_url, store_path) = if let Some(server) = args.server {
        (server, None)
    } else {
        // Try to get from current repo
        let workspace_command = command_helper.workspace_helper(ui)?;
        let store_path = workspace_command.repo_path().join("store");
        let config = JjhubConfig::load(&store_path).map_err(|e| {
            user_error(format!(
                "Not in a jjhub repository and no --server specified: {}",
                e
            ))
        })?;
        (config.base_url, Some(store_path))
    };

    // Get username
    let username = if let Some(u) = args.username {
        u
    } else {
        write!(ui.status(), "Username: ")?;
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| user_error(format!("Failed to read username: {}", e)))?;
        input.trim().to_string()
    };

    if username.is_empty() {
        return Err(user_error("Username cannot be empty"));
    }

    // Get password (hidden input)
    let password = rpassword::prompt_password("Password: ")
        .map_err(|e| user_error(format!("Failed to read password: {}", e)))?;

    if password.is_empty() {
        return Err(user_error("Password cannot be empty"));
    }

    // Attempt to authenticate
    writeln!(ui.status(), "Authenticating...")?;
    let client = JjhubClient::new(&server_url);
    let token = client
        .login(&username, &password)
        .map_err(|e| user_error(format!("Authentication failed: {}", e)))?;

    // Save token to config if we're in a repo
    if let Some(store_path) = store_path {
        let mut config = JjhubConfig::load(&store_path)
            .map_err(|e| user_error(format!("Failed to load config: {}", e)))?;
        config.set_token(Some(token.clone()));
        config
            .save(&store_path)
            .map_err(|e| user_error(format!("Failed to save token: {}", e)))?;
        writeln!(
            ui.status(),
            "Logged in as {} (token saved to repository)",
            username
        )?;
    } else {
        // Print token if not in a repo (user can manually save it)
        writeln!(ui.status(), "Logged in as {}", username)?;
        writeln!(ui.status(), "Token: {}", token)?;
        writeln!(
            ui.hint_default(),
            "Run this command from within a jjhub repository to save the token automatically"
        )?;
    }

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
