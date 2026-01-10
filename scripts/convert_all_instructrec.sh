#!/bin/bash
# Convert all InstructRec datasets from iAgent format to MemRec format

IAGENT_DATA_DIR="data/iagent"
OUTPUT_BASE_DIR="data/processed"

echo "════════════════════════════════════════════════════════════════"
echo "  Convert all InstructRec datasets to MemRec format"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Books
echo "Processing Amazon Books..."
python scripts/convert_iagent_to_memrec.py \
    --input ${IAGENT_DATA_DIR}/booksAll_recagent.pkl \
    --mapping ${IAGENT_DATA_DIR}/combined_books_asin_mapping.csv \
    --output ${OUTPUT_BASE_DIR}/instructrec-books \
    --domain books \
    --mode full

echo ""
echo "────────────────────────────────────────────────────────────────"
echo ""

# 2. MovieTV
echo "Processing Amazon MovieTV..."
python scripts/convert_iagent_to_memrec.py \
    --input ${IAGENT_DATA_DIR}/movietvAll_recagent.pkl \
    --mapping ${IAGENT_DATA_DIR}/combined_movietv_asin_mapping.csv \
    --output ${OUTPUT_BASE_DIR}/instructrec-movietv \
    --domain movietv \
    --mode full

echo ""
echo "────────────────────────────────────────────────────────────────"
echo ""

# 3. GoodReads
echo "Processing GoodReads..."
python scripts/convert_iagent_to_memrec.py \
    --input ${IAGENT_DATA_DIR}/readsAll_recagent.pkl \
    --mapping ${IAGENT_DATA_DIR}/combined_reads_asin_mapping.csv \
    --output ${OUTPUT_BASE_DIR}/instructrec-goodreads \
    --domain goodreads \
    --mode full

echo ""
echo "────────────────────────────────────────────────────────────────"
echo ""

# 4. Yelp
echo "Processing Yelp..."
python scripts/convert_iagent_to_memrec.py \
    --input ${IAGENT_DATA_DIR}/yelpAll_recagent.pkl \
    --mapping ${IAGENT_DATA_DIR}/combined_yelp_asin_mapping.csv \
    --output ${OUTPUT_BASE_DIR}/instructrec-yelp \
    --domain yelp \
    --mode full

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All datasets converted successfully!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Conversion results:"
for domain in books movietv goodreads yelp; do
    dir="${OUTPUT_BASE_DIR}/instructrec-${domain}"
    if [ -d "$dir" ]; then
        echo ""
        echo "  instructrec-${domain}:"
        if [ -f "$dir/statistics.txt" ]; then
            cat "$dir/statistics.txt" | head -n 4 | sed 's/^/    /'
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"

