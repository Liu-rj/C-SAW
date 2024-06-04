
# ./streaming.bin web-google web-google/beg.bin web-google/csr.bin 10 128 $1 $2 $3 $4 $5

# Web-Google
## deepwalk
./streaming.bin web-google web-google/beg_pos.bin web-google/csr.bin 10 128 4000 0 1 2000 1
./streaming.bin web-google web-google/beg_pos.bin web-google/csr.bin 20 128 10000 0 1 100 1
./streaming.bin web-google web-google/beg_pos.bin web-google/csr.bin 20 128 10000 0 1 2000 1
## graphsage
# ./streaming.bin web-google web-google/beg_pos.bin web-google/csr.bin 10 128 1024 0 10 2 1
# ./streaming.bin web-google web-google/beg_pos.bin web-google/csr.bin 20 128 1024 0 10 3 1
# ./streaming.bin web-google web-google/beg_pos.bin web-google/csr.bin 20 128 4096 0 10 3 1


# LiveJournal
## deepwalk
./streaming.bin livejournal livejournal/beg_pos.bin livejournal/csr.bin 10 128 4000 0 1 2000 1
./streaming.bin livejournal livejournal/beg_pos.bin livejournal/csr.bin 20 128 10000 0 1 100 1
./streaming.bin livejournal livejournal/beg_pos.bin livejournal/csr.bin 20 128 10000 0 1 2000 1

## graphsage
# ./streaming.bin livejournal livejournal/beg_pos.bin livejournal/csr.bin 10 128 1024 0 10 2 1
# ./streaming.bin livejournal livejournal/beg_pos.bin livejournal/csr.bin 20 128 1024 0 10 3 1
# ./streaming.bin livejournal livejournal/beg_pos.bin livejournal/csr.bin 20 128 4096 0 10 3 1

# ./streaming.bin friendster friendster/com-friendster.ungraph.txt_beg_pos.bin friendster/com-friendster.ungraph.txt_csr.bin 20 128 4000 0 1 2000 1
# ./streaming.bin web-google web-google/beg_pos.bin web-google/csr.bin 1 32 4000 0 1 5 1
# ./streaming.bin livejournal livejournal/beg_pos.bin livejournal/csr.bin 10 128 4000 0 1 100 1
