#!/bin/bash
# Launch script for Plant Diagnostic System v2.0 (demo_v5.py)

echo "============================================================"
echo "🌿 Plant Diagnostic System v2.0"
echo "============================================================"
echo ""

# Default settings
CFG_PATH="eval_configs/minigptv2_eval.yaml"
GPU_ID=0
SHARE_FLAG=""
RESNET_ANCHOR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --share)
            SHARE_FLAG="--share"
            echo "✓ Public share link enabled"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            echo "✓ Using GPU: $GPU_ID"
            shift 2
            ;;
        --cfg-path)
            CFG_PATH="$2"
            echo "✓ Config: $CFG_PATH"
            shift 2
            ;;
        --resnet-anchor)
            RESNET_ANCHOR="--resnet-anchor"
            echo "✓ ResNet anchor enabled"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --share              Create public share link (accessible from anywhere)"
            echo "  --gpu ID             GPU device ID (default: 0)"
            echo "  --cfg-path PATH      Path to config file (default: eval_configs/minigptv2_eval.yaml)"
            echo "  --resnet-anchor      Enable ResNet anchor for diagnosis"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Basic local usage"
            echo "  $0 --share                      # With public share link"
            echo "  $0 --share --resnet-anchor      # Full features with share"
            echo "  $0 --gpu 1 --share              # Use GPU 1 with share"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  - Config file: $CFG_PATH"
echo "  - GPU ID: $GPU_ID"
if [ -n "$SHARE_FLAG" ]; then
    echo "  - Public share: ENABLED"
else
    echo "  - Public share: DISABLED (use --share to enable)"
fi
if [ -n "$RESNET_ANCHOR" ]; then
    echo "  - ResNet anchor: ENABLED"
fi
echo ""

# Check if config file exists
if [ ! -f "$CFG_PATH" ]; then
    echo "❌ Error: Config file not found at $CFG_PATH"
    exit 1
fi

# Launch the demo
echo "🚀 Launching Plant Diagnostic System..."
echo ""

python3 demo_v5.py \
    --cfg-path "$CFG_PATH" \
    --gpu-id "$GPU_ID" \
    $SHARE_FLAG \
    $RESNET_ANCHOR
