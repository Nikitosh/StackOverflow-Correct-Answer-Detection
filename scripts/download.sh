wget -P SO_data "https://archive.org/download/stackexchange/$1.7z"
7z e -oSO_data "SO_data/$1.7z"
rm -rf "SO_data/$1.7z" SO_data/Badges.xml SO_data/Comments.xml SO_data/PostHistory.xml SO_data/PostLinks.xml SO_data/Tags.xml SO_data/Users.xml SO_data/Votes.xml
mv SO_data/Posts.xml "SO_data/$2.xml"
python3 preprocess_xml.py --xml_path "SO_data/$2.xml"
rm "SO_data/$2.xml"