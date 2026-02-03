$bucket = "99v3pnjmjj"
$endpoint = "https://s3api-us-nc-2.runpod.io"

# Verify AWS CLI check
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "Error: AWS CLI is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Uploading SFT Data..." -ForegroundColor Cyan
aws s3 cp "E:\Ridwanur\Documents\Y3 FYM\sft_scripts\sft_data\sft_train.jsonl" "s3://$bucket/sft_data/sft_train.jsonl" --endpoint-url $endpoint --region us-nc-2
aws s3 cp "E:\Ridwanur\Documents\Y3 FYM\sft_scripts\sft_data\sft_val.jsonl" "s3://$bucket/sft_data/sft_val.jsonl" --endpoint-url $endpoint --region us-nc-2

Write-Host "Uploading Checkpoints..." -ForegroundColor Cyan
aws s3 cp "E:\Ridwanur\Documents\Y3 FYM\Checkpoints\hybrid_19072.pt" "s3://$bucket/checkpoints/hybrid_19072.pt" --endpoint-url $endpoint --region us-nc-2
aws s3 cp "E:\Ridwanur\Documents\Y3 FYM\Checkpoints\mla_19072.pt" "s3://$bucket/checkpoints/mla_19072.pt" --endpoint-url $endpoint --region us-nc-2

Write-Host "Upload Complete!" -ForegroundColor Green
