from huggingface_hub import snapshot_download, hf_hub_download
import os

def download_all_videos():
    """Download all videos from the dataset"""
    local_dir = "./assets/"
    
    # Download only the videos folder
    snapshot_download(
        repo_id="tomg-group-umd/argus",
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns="videos/*.mp4",  # Only download files in videos folder
        cache_dir=None,  # Don't use cache, download directly
        local_dir_use_symlinks=False  # Create actual files, not symlinks
    )
    
    ## rename "assets/videos" to "assets/clips"
    videos_dir = os.path.join(local_dir, "videos")
    clips_dir = os.path.join(local_dir, "clips")
    os.rename(videos_dir, clips_dir)
    
    ## check total number of videos downloaded
    total_videos = len(os.listdir(clips_dir))
    
    print(f"{total_videos} videos downloaded to {local_dir}/clips/")

if __name__ == "__main__":
    print("\nDownloading all videos...")
    download_all_videos()