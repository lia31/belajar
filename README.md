Program ini merupakan program klasifikasi judul skripsi berdasarkan peminatan
pada Program Studi Pendidikan Teknik Informatika dan Komputer Universitas Negeri
Jakarta.

Dataset yang didapatkan dari tahun 2010 sampai dengan tahun 2013 semester ganjil
yang dapat dilihat pada folder dataset. dataset tersebut berasal dari judul
skripsi yang sudah direvisi dan diterima di Universitas.

Peneliti menggunakan algoritma klasifikasi terbaik menurut ICDM 2006, yaitu 
algoritma K-Nearest Neighbor, Naive Bayes Classifier, dan Support Vector Machine
.

Klasifikasi terhadap dokumen teks pendek seperti judul skripsi mahasiswa 
memiliki kesulitan tersendiri daripada dokumen teks panjang karena semakin 
sedikit kata semakin sulit diklasifikasi. Sehingga tujuan dari penelitian ini
adalah untuk mengetahui algoritma yang paling efisien untuk mengklasifikasi 
judul skripsi.

Penelitian ini terdiri dari beberapa tahap yaitu pengumpulan data, pengelompokan
data melalui angket oleh dosen ahli, pre-processing text, pembobotan kata 
menggunakan vector space model dan tf-idf, evaluasi dengan kfold cross validation, 
klasifikasi menggunakan k-nearest neighbor, naïve bayes classifier, dan support
vector machine, dan analisis dengan confusion matrix.