set -e

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..

GTEST_DIR="${TOP_DIR}/opensource/googletest"
if [ ! -d "$GTEST_DIR" ]; then
    cd ${TOP_DIR}/opensource
    git clone https://gitcode.com/GitHub_Trending/go/googletest.git googletest -b release-1.12.0
else
    echo "opensource/googletest already exists. no need to download. exit."
fi

SECUREC_DIR="${TOP_DIR}/platform/securec"
if [ ! -d "$SECUREC_DIR" ]; then
    cd ${TOP_DIR}/platform
    git clone https://gitcode.com/openeuler/libboundscheck.git securec -b v1.1.16
else
    echo "platform/securec already exists. no need to download. exit."
fi

JSON_PATH="${TOP_DIR}/opensource/json"
if [ ! -d "$JSON_PATH" ]; then
    cd ${TOP_DIR}/opensource
    git clone https://gitcode.com/GitHub_Trending/js/json.git json -b v3.11.3
else
    echo "opensource/json already exists. no need to download. exit."
fi

SQLITE_DIR="${TOP_DIR}/opensource/sqlite3"
SQLITE_VERSION="3450300"  # corresponds to 3.45.3
SQLITE_URL="https://www.sqlite.org/2024/sqlite-amalgamation-${SQLITE_VERSION}.zip"

if [ ! -d "$SQLITE_DIR" ] || [ ! -f "${SQLITE_DIR}/sqlite3.c" ]; then
    echo "Downloading SQLite amalgamation v${SQLITE_VERSION}..."
    mkdir -p "${TOP_DIR}/opensource"
    cd "${TOP_DIR}/opensource"

    # Try wget first, fall back to curl
    if command -v wget >/dev/null 2>&1; then
        wget -O sqlite3.zip "${SQLITE_URL}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o sqlite3.zip "${SQLITE_URL}"
    else
        echo "Error: Neither wget nor curl is available. Please install one." >&2
        exit 1
    fi

    # Unzip and rename
    unzip -q sqlite3.zip
    # The extracted folder is named like: sqlite-amalgamation-3450300
    EXTRACTED_DIR="sqlite-amalgamation-${SQLITE_VERSION}"
    mkdir -p sqlite3
    if [ -d "$EXTRACTED_DIR" ]; then
        cp "$EXTRACTED_DIR"/sqlite3.c sqlite3/
        cp "$EXTRACTED_DIR"/sqlite3.h sqlite3/
    else
        echo "Error: Failed to extract SQLite amalgamation." >&2
        exit 1
    fi

    rm -f sqlite3.zip
    rm -rf "$EXTRACTED_DIR"
    echo "SQLite downloaded to opensource/sqlite3/"
else
    echo "opensource/sqlite3 already exists. Skipping."
fi