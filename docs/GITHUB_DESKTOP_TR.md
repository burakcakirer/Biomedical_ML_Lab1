# GitHub Desktop ile Windows üzerinden yükleme

**Yerel proje konumu:** `D:\GitHub\Biomedical_ML_Lab1`

1. İndirilen ana proje ZIP dosyasını açın. ZIP'in içindeki **Biomedical_ML_Lab1** klasörünü `D:\GitHub` klasörüne kopyalayın. Nihai yol `D:\GitHub\Biomedical_ML_Lab1\README.md` olmalıdır. İkinci bir iç içe `Biomedical_ML_Lab1` klasörü oluşmamalı.
2. **GitHub Desktop → File → Add local repository → Choose** ile `D:\GitHub\Biomedical_ML_Lab1` klasörünü seçin.
3. “This directory does not appear to be a Git repository” uyarısı çıkarsa **Create a repository** seçeneğine basın. Repository name olarak `Biomedical_ML_Lab1` belirleyin; local path olarak `D:\GitHub` seçili olmalı (çünkü yeni repo klasörü Desktop tarafından bu yolun içine açılır). Halihazırdaki dolu klasöre farklı bir boş klasör oluşturmayın.
4. İlk kez oluştururken **Changes** ekranında dosyalar görünmelidir. Değişiklik varsa commit özeti `Initial commit: Biomedical ML Lab 1` yazıp **Commit to main** seçin. 0 changed files görüntülerseniz **History** sekmesinde ilk commit'in var olup olmadığını kontrol edin; README dahil dosyalar mevcutsa ve ilk commit varsa normaldir.
5. **Publish repository** butonuna basın, hesabı ve repository adını seçin. Herkesin görebilmesi için **Keep this code private** işaretini kaldırın. **Publish Repository** deyin.
6. İşlem bitince **Repository → View on GitHub** ile internet sitesinden kontrol edin. README, `src/`, `results/` ve `report/` klasörleri açılmalıdır. Raporun PDF dosyasını GitHub üzerinden açarak tekrar doğrulayın.
7. Daha sonra dosyaları güncellerseniz GitHub Desktop **Changes → Summary → Commit to main → Push origin** adımlarını izleyin.

**Not:** GitHub Desktop proje dosyalarını kendi başına indirmez veya içeriğini değiştirmez; ZIP'ten çıkardığınız dosyaların doğru klasörde olduğundan emin olun. `D:\GitHub` klasörünü değil, içindeki proje klasörünü repo olarak açın.
