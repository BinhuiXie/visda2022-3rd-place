
set -e  # exit when any command fails
mkdir -p data/
cd data/
# ZeroWaste V1
wget http://csr.bu.edu/ftp/recycle/zerowaste-f.zip 
mkdir zerowaste-f
unzip zerowaste-f.zip zerowaste-f
rm zerowaste-f.zip

wget http://csr.bu.edu/ftp/recycle/visda-2022/zerowaste-v2-train.zip
mkdir zerowaste-v2
unzip zerowaste-v2-train.zip zerowaste-v2
rm zerowaste-f.zip

cd ../

