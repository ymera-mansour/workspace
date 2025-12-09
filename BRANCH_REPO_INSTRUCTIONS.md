# Create a new repository from this branch only

The current branch is `copilot/create-new-repo-branch`. Use the steps below to publish only this branch into a fresh repository.

## Option A: Preserve this branch's history
1. Clone just this branch:  
   ```bash
   git clone --branch copilot/create-new-repo-branch --single-branch <CURRENT_REPO_URL> new-repo
   ```
2. Enter the clone and point it at the new remote:  
   ```bash
   cd new-repo
   git remote remove origin
   git remote add origin <NEW_REPO_URL>
   ```
3. Push the branch to the new repository:  
   ```bash
   git push -u origin copilot/create-new-repo-branch
   ```
4. In the new repo, set this branch as the default in your hosting UI (only needed if you want to keep this branch name; otherwise rename to something like `main`).

## Option B: Start a clean history (single initial commit)
1. Clone just this branch:  
   ```bash
   git clone --branch copilot/create-new-repo-branch --single-branch <CURRENT_REPO_URL> new-repo
   cd new-repo
   ```
2. Strip existing history and reinitialize:  
   ```bash
   rm -rf .git
   git init
   ```
3. Create the initial commit and push as `main` (or any name you prefer):  
   ```bash
   git add .
   git commit -m "Initial import from copilot/create-new-repo-branch"
   git branch -M main
   git remote add origin <NEW_REPO_URL>
   git push -u origin main
   ```

Pick Option A if you need to keep the branch's commit history; pick Option B if you want a clean, single-commit starting point.
