#build and install find orb
mkdir -p find_orb
cd find_orb
git clone https://github.com/Bill-Gray/lunar.git
cd lunar
git checkout 005ccaa469b32abc0df84b512d52e8b1c80efbda
cd ..
git clone https://github.com/Bill-Gray/sat_code.git
cd sat_code
git checkout 66afa0860da796a4ff4b4557531fb4e6ae44a095
cd ..
git clone https://github.com/Bill-Gray/jpl_eph.git
cd jpl_eph
git checkout 0c2782e86e42df69a55c2b2db0b72a40312c79c0
cd ..
git clone https://github.com/Bill-Gray/find_orb.git
cd find_orb
git checkout f574af87ed11f5ec5b69ef8125e8b539de6d6645
cd ..
git clone https://github.com/Bill-Gray/miscell.git
cd miscell
git checkout f4565afdf9d0324e798527f837e0814f8de0abe0
cd ..
cd jpl_eph
make libjpl.a
make install
cd ../lunar
make
make integrat
make install
cd ../sat_code
make sat_id
make install
cd ../find_orb
make
make install
cd ../.find_orb
wget ftp://ssd.jpl.nasa.gov/pub/eph/planets/Linux/de440t/linux_p1550p2650.440t
cd -
cd ../..
