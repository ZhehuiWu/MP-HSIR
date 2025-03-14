The download addresses of each data set are as follows:

Natural scene datasets :

**ICVL :** https://icvl.cs.bgu.ac.il/hyperspectral/, The specific division of the data set is shown in data/ICVL_train_list.txt and data/ICVL_test_list.txt. 

**ARAD :** https://codalab.lisn.upsaclay.fr/competitions/721, The validation dataset in the official website is used as the test dataset here. 


Remote sensing datasets :

**Xiong'an :** http://www.hrs-cas.com/a/share/shujuchanpin/2019/0501/1049.html 

**WDC :** https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html

**PaviaU :** https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene 

**PaviaC :** https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene 

**Houston :** https://www.grss-ieee.org/community/technical-committees/data-fusion/2013-ieee-grss-data-fusion-contest/ 

**Chikusei :** https://naotoyokoya.com/Download.html

**Eagle:** https://figshare.com/articles/dataset/Main_zip/2007723/3?file=3576183 

**Berlin :** https://dataservices.gfz-potsdam.de/enmap/showshort.php?id=escidoc:1480925 

**Urban :** https://rslab.ut.ac.ir/data 

**APEX :** https://apex-esa.org/en/data/free-data-cubes

**EO-1 :** https://earthexplorer.usgs.gov/

The specific segmentation strategy of the remote sensing dataset can be found in utils/mat_data.py.

The method of making the lmdb training set can be found in utils/lmdb_patch.py.