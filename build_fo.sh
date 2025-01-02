#build and install find orb
mkdir -p find_orb
cd find_orb
git clone https://github.com/Bill-Gray/lunar.git
git clone https://github.com/Bill-Gray/sat_code.git
git clone https://github.com/Bill-Gray/jpl_eph.git
git clone https://github.com/Bill-Gray/find_orb.git
git clone https://github.com/Bill-Gray/miscell.git
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
wget ftp://ssd.jpl.nasa.gov/pub/eph/planets/Linux/de430t/linux_p1550p2650.430t
cd -
cd ../..
