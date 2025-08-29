use std::{
    future::Future,
    path::{Component, Path, PathBuf},
};

use color_eyre::eyre::{OptionExt, eyre};
use hxd::AsHexd;
use itertools::Itertools;
use log::info;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::AsyncReadExt;
use tokio_stream::{StreamExt, wrappers::ReadDirStream};

use crate::{error::AgentyError, tool::Tool};

pub fn sanitize_join_relative_path(cwd: &Path, rpath: &Path) -> Result<PathBuf, String> {
    if rpath.is_absolute() {
        return Err(format!("{:?} is an absolute path", rpath));
    }
    if rpath.components().any(|t| t == Component::ParentDir) {
        return Err(format!("{:?} contains '..'", rpath));
    }

    Ok(cwd.join(rpath))
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct ReadFileToolArgs {
    pub file_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ReadFileTool {
    pub cwd: PathBuf,
}

impl ReadFileTool {
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }

    pub async fn read_file(&self, file_path: PathBuf) -> Result<String, AgentyError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &file_path) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        match tokio::fs::metadata(&target_path).await {
            Ok(meta) => {
                if meta.is_dir() {
                    return Ok(format!("Path {:?} is a directory", &target_path));
                }
            }
            Err(e) => {
                return Ok(format!(
                    "Fail to get metadata of {:?} due to {}",
                    &target_path, e
                ));
            }
        };
        let mut fp = match tokio::fs::File::open(&target_path).await {
            Ok(fp) => fp,
            Err(e) => return Ok(format!("Fail to open {:?} due to {}", &target_path, e)),
        };

        let mut buf = vec![];
        fp.read_to_end(&mut buf).await?;

        let buf = if buf.len() >= 8192 {
            // too long and cutoff
            buf[0..8192].to_vec()
        } else {
            buf
        };

        match String::from_utf8(buf) {
            Ok(s) => Ok(s),
            Err(e) => Ok(e.into_bytes().hexd().dump_to::<String>()),
        }
    }
}

impl Tool for ReadFileTool {
    type ARGUMENTS = ReadFileToolArgs;
    const NAME: &str = "read_file";
    const DESCRIPTION: Option<&str> = Some(
        "Read file contents of the path `file_path`. The result will be hexdump if the file is a binary file.",
    );

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, AgentyError>> + Send {
        self.read_file(arguments.file_path)
    }
}

pub fn list_files(cwd: &Path, fpaths: Vec<PathBuf>) -> Result<Vec<String>, AgentyError> {
    let mut lns = vec![];
    let cwd = cwd.canonicalize()?;
    for fp in fpaths {
        let meta = fp.metadata()?;
        let ln = format!(
            "{:?}\t{}\t{}",
            fp.canonicalize()?
                .strip_prefix(&cwd)
                .expect(&format!("{:?} not relative to {:?}?!", &fp, cwd)),
            if meta.is_dir() {
                "directory"
            } else if meta.is_file() {
                "file"
            } else if meta.is_symlink() {
                "symlink"
            } else {
                ""
            },
            meta.len()
        );
        lns.push(ln);
    }
    Ok(lns)
}

#[derive(Deserialize, JsonSchema)]
pub struct ListDirectoryToolArgs {
    pub relative_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ListDirectoryTool {
    pub cwd: PathBuf,
}

impl ListDirectoryTool {
    pub fn new_root(path: PathBuf) -> Self {
        Self { cwd: path }
    }
    pub async fn list_directory(&self, relative_path: PathBuf) -> Result<String, AgentyError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &relative_path) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        if !target_path.is_dir() {
            return Ok(format!("{:?} is not a directory", &target_path));
        }

        let mut st = ReadDirStream::new(tokio::fs::read_dir(&target_path).await?);
        let mut items = vec![];
        while let Some(ent) = st.next().await {
            let ent = ent?;
            items.push(ent.path());
        }
        let lns = list_files(&self.cwd, items)?;
        Ok(format!(
            "The contents of folder {:?} is:\nname\ttype\tsize\n{}",
            &relative_path,
            lns.into_iter().join("\n")
        ))
    }
}

impl Tool for ListDirectoryTool {
    type ARGUMENTS = ListDirectoryToolArgs;
    const NAME: &str = "list_dir";
    const DESCRIPTION: Option<&str> = Some(
        "List a given directory entries. '.' is allowed to list entries of the root directory but '..' is not allowed to avoid path traversal. Absolute path is not allowed and you shall always use relative path to the root directory.",
    );

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, AgentyError>> + Send {
        self.list_directory(arguments.relative_path)
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct FindFileArgs {
    pub directory: PathBuf,
    pub file_name_pattern: String,
}

#[derive(Debug, Clone)]
pub struct FindFileTool {
    pub cwd: PathBuf,
}

impl FindFileTool {
    pub fn new(path: PathBuf) -> Self {
        Self { cwd: path }
    }
    pub fn find_file(
        cwd: PathBuf,
        directory: PathBuf,
        pattern: String,
    ) -> Result<String, AgentyError> {
        let re = match glob::Pattern::new(&pattern) {
            Ok(re) => re,
            Err(e) => return Ok(format!("Fail to compile the glob pattern due to {}", e)),
        };

        let target_path = match sanitize_join_relative_path(&cwd, &directory) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        if !target_path.is_dir() {
            return Ok(format!("{:?} is not a directory", &target_path));
        }

        let mut items = vec![];
        for ent in walkdir::WalkDir::new(&target_path) {
            let ent = ent?;
            let fname = ent
                .file_name()
                .to_str()
                .ok_or_eyre(eyre!("non-utf8 fname ignored {:?}", &ent))?;
            if re.matches(&fname) {
                items.push(ent.path().to_path_buf());
            }
        }
        let lns = list_files(&cwd, items)?;
        Ok(format!(
            "The files found under directory {:?} with given pattern {} are:\n{}",
            &directory,
            &pattern,
            lns.into_iter().join("\n")
        ))
    }
}

impl Tool for FindFileTool {
    type ARGUMENTS = FindFileArgs;
    const NAME: &str = "find_file";
    const DESCRIPTION: Option<&str> = Some(
        "Find files with names having the given glob pattern under the given directory. For example, use '*.c' to find all C source files. For directory, note '.' is allowed to list entries of the root directory but '..' is not allowed to avoid path traversal. Absolute path is not allowed and you shall always use relative path to the root directory.",
    );

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, AgentyError>> + Send {
        let cwd = self.cwd.clone();
        async move {
            tokio::task::spawn_blocking(move || {
                Self::find_file(cwd, arguments.directory, arguments.file_name_pattern)
            })
            .await
            .expect("fail to join")
        }
    }
}
