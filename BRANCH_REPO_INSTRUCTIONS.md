# Create a new repository from this branch only

Target branch: `copilot/review-gemini-models-integration`  
Desired new repository name: `builder`  
Use the steps below to publish only this branch into a fresh repository.

## Option A: Preserve this branch's history
1. Clone just this branch:  
   ```bash
   git clone --branch copilot/review-gemini-models-integration --single-branch <SOURCE_REPO_URL> builder
   ```
2. Enter the clone and point it at the new remote:  
   ```bash
   cd builder
   git remote remove origin
   git remote add origin <NEW_REPO_URL>
   ```
3. Push the branch to the new repository:  
   ```bash
   git push -u origin copilot/review-gemini-models-integration
   ```
4. In the new repo, set this branch as the default in your hosting UI (use your provider's UI, e.g., GitHub or GitLab project settings). If you prefer to call it `main`, rename locally and push that instead:  
   ```bash
   git branch -M main
   git push -u origin main
   ```

## Option B: Start a clean history (single initial commit)
1. Clone just this branch:  
   ```bash
   git clone --branch copilot/review-gemini-models-integration --single-branch <SOURCE_REPO_URL> builder
   cd builder
   ```
2. Strip existing history and reinitialize:  
   ```bash
   rm -rf .git
   git init
   ```
3. Create the initial commit and push as `main` (or any name you prefer):  
   ```bash
   git add .
   git commit -m "Initial import from copilot/review-gemini-models-integration"
   git branch -M main
   git remote add origin <NEW_REPO_URL>
   git push -u origin main
   ```

Pick Option A to retain full history and authorship; pick Option B when you want a clean, single-commit starting point (for example, when shipping a product snapshot or avoiding sensitive history).

Security note: Option B does not scrub sensitive data from any clones that already exist. If the original repository may contain sensitive history, use proper history-rewrite tools (e.g., `git filter-repo`) and coordinate with all copies.
