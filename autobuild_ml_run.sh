EXPOSED_PORT=12463

SERVER_IP=213.173.108.41
REMOTE_FOLDER="workspace/"
LOCAL_FOLDER="/Users/meguka/GIT/interactive-bees/"
LYN_APP_PATH="/Applications/Lyn.app"

# Install rsync on remote server if not installed
ssh -o StrictHostKeyChecking=no -p $EXPOSED_PORT root@$SERVER_IP \
    'if ! command -v rsync &> /dev/null; then apt-get update && apt-get install -y rsync; fi'

# Sync files to remote server
rsync -l -r -v -t --no-perms --delete \
    --exclude 'data' \
    --exclude 'log.txt' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    -e "ssh -p $EXPOSED_PORT" \
    $LOCAL_FOLDER \
    root@$SERVER_IP:/$REMOTE_FOLDER

# Run code on remote server and capture image_name
image_name=$(ssh -p $EXPOSED_PORT root@$SERVER_IP "cd /$REMOTE_FOLDER && python ml_run.py" | tail -n 1)

# Sync files back from remote server
rsync -l -r -v -t --no-perms --update \
    --exclude 'data' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    -e "ssh -p $EXPOSED_PORT" \
    root@$SERVER_IP:/$REMOTE_FOLDER/ \
    $LOCAL_FOLDER

echo "Captured image name: $image_name"
open -g -a "$LYN_APP_PATH" "$image_name"
