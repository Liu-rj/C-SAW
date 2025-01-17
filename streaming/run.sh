
# ./streaming.bin web-google web-google/beg.bin web-google/csr.bin 10 128 $1 $2 $3 $4 $5

# Deepwalk
## Web-Google
./streaming.bin web-google dataset/web-google/beg_pos.bin dataset/web-google/csr.bin 10 128 4000 0 1 2000 1
./streaming.bin web-google dataset/web-google/beg_pos.bin dataset/web-google/csr.bin 20 128 10000 0 1 100 1
./streaming.bin web-google dataset/web-google/beg_pos.bin dataset/web-google/csr.bin 20 128 10000 0 1 2000 1
## LiveJournal
./streaming.bin livejournal dataset/livejournal/beg_pos.bin dataset/livejournal/csr.bin 10 128 4000 0 1 2000 1
./streaming.bin livejournal dataset/livejournal/beg_pos.bin dataset/livejournal/csr.bin 20 128 10000 0 1 100 1
./streaming.bin livejournal dataset/livejournal/beg_pos.bin dataset/livejournal/csr.bin 20 128 10000 0 1 2000 1
## Reddit
./streaming.bin reddit dataset/reddit/beg_pos.bin dataset/reddit/csr.bin 10 128 4000 0 1 2000 1
./streaming.bin reddit dataset/reddit/beg_pos.bin dataset/reddit/csr.bin 20 128 10000 0 1 100 1
./streaming.bin reddit dataset/reddit/beg_pos.bin dataset/reddit/csr.bin 20 128 10000 0 1 2000 1
## Ogbn-products
./streaming.bin ogbn_products dataset/ogbn_products/beg_pos.bin dataset/ogbn_products/csr.bin 10 128 4000 0 1 2000 1
./streaming.bin ogbn_products dataset/ogbn_products/beg_pos.bin dataset/ogbn_products/csr.bin 20 128 10000 0 1 100 1
./streaming.bin ogbn_products dataset/ogbn_products/beg_pos.bin dataset/ogbn_products/csr.bin 20 128 10000 0 1 2000 1
## Friendster
./streaming.bin friendster dataset/friendster/beg_pos.bin dataset/friendster/csr.bin 10 128 4000 0 1 2000 1
./streaming.bin friendster dataset/friendster/beg_pos.bin dataset/friendster/csr.bin 20 128 10000 0 1 100 1
./streaming.bin friendster dataset/friendster/beg_pos.bin dataset/friendster/csr.bin 20 128 10000 0 1 2000 1


# GraphSAGE
## Web-Google
./streaming.bin web-google dataset/web-google/beg_pos.bin dataset/web-google/csr.bin 10 128 1024 0 10 2 1
./streaming.bin web-google dataset/web-google/beg_pos.bin dataset/web-google/csr.bin 20 128 1024 0 10 3 1
./streaming.bin web-google dataset/web-google/beg_pos.bin dataset/web-google/csr.bin 20 128 4096 0 10 3 1
## LiveJournal
./streaming.bin livejournal dataset/livejournal/beg_pos.bin dataset/livejournal/csr.bin 10 128 1024 0 10 2 1
./streaming.bin livejournal dataset/livejournal/beg_pos.bin dataset/livejournal/csr.bin 20 128 1024 0 10 3 1
./streaming.bin livejournal dataset/livejournal/beg_pos.bin dataset/livejournal/csr.bin 20 128 4096 0 10 3 1
## Reddit
./streaming.bin reddit dataset/reddit/beg_pos.bin dataset/reddit/csr.bin 10 128 1024 0 10 2 1
./streaming.bin reddit dataset/reddit/beg_pos.bin dataset/reddit/csr.bin 20 128 1024 0 10 3 1
./streaming.bin reddit dataset/reddit/beg_pos.bin dataset/reddit/csr.bin 20 128 4096 0 10 3 1
## Obgn-products
./streaming.bin ogbn_products dataset/ogbn_products/beg_pos.bin dataset/ogbn_products/csr.bin 10 128 1024 0 10 2 1
./streaming.bin ogbn_products dataset/ogbn_products/beg_pos.bin dataset/ogbn_products/csr.bin 20 128 1024 0 10 3 1
./streaming.bin ogbn_products dataset/ogbn_products/beg_pos.bin dataset/ogbn_products/csr.bin 20 128 4096 0 10 3 1
## Friendster
./streaming.bin friendster dataset/friendster/beg_pos.bin dataset/friendster/csr.bin 10 128 1024 0 10 2 1
./streaming.bin friendster dataset/friendster/beg_pos.bin dataset/friendster/csr.bin 20 128 1024 0 10 3 1
./streaming.bin friendster dataset/friendster/beg_pos.bin dataset/friendster/csr.bin 20 128 4096 0 10 3 1


# ./streaming.bin friendster dataset/friendster/com-friendster.ungraph.txt_beg_pos.bin dataset/friendster/com-friendster.ungraph.txt_csr.bin 20 128 4000 0 1 2000 1
# ./streaming.bin web-google dataset/web-google/beg_pos.bin dataset/web-google/csr.bin 1 32 4000 0 1 5 1
# ./streaming.bin livejournal dataset/livejournal/beg_pos.bin dataset/livejournal/csr.bin 10 128 4000 0 1 100 1
