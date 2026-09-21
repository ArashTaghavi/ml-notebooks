# Measured clustering results

Workbook SHA-256: `ad3af9555a4499bac64031eb8980d4f389c4a94dcddbc6976f5915d6879c36a3`
Search: 1407 configurations; 760 resampling fits.
Primary objective: customer spending behavior (income + spending score).

- **all_features**: KMeans, {"n_clusters": 5}; representation `all_features/standard/cat=0.25`; k=5, reference silhouette=0.3503, coverage=100.0%, median stability ARI=0.9842, 10th percentile=0.6311.
- **behavior**: KMeans, {"n_clusters": 5}; representation `behavior/standard/cat=0.5`; k=5, reference silhouette=0.5547, coverage=100.0%, median stability ARI=0.9917, 10th percentile=0.9300.
- **behavior_age**: KMeans, {"n_clusters": 6}; representation `behavior_age/robust/cat=0.5`; k=6, reference silhouette=0.4292, coverage=100.0%, median stability ARI=0.9800, 10th percentile=0.9469.

## Interpretation / تفسیر نتایج

KMeans settings in this run: init=k-means++, n_init=30, random_state=42, max_iter=300, algorithm=lloyd. Numeric imputation uses the median; final fitted scalers and categorical encoders are stored in each bundle.

برای رفتار خرید، درآمد و امتیاز خرید ورودی مدل‌اند. سن و مشخصات جمعیت‌شناختی در این مدل برای تفسیر خوشه‌ها استفاده می‌شوند. در مدل all_features تمام شش ویژگی غیرشناسه وارد مدل می‌شوند، با وزن دسته‌ای نمایش‌داده‌شده در جدول؛ این وزن یک انتخاب مدل‌سازی است، نه اهمیت علّی ویژگی‌ها.

حساسیت مدل همهٔ ویژگی‌ها به حذف تحصیلات و وضعیت تأهل: ARI=1.0000. مقدار یک یعنی حذف این دو ستون در همین تنظیمات، عضویت خوشه‌ها را تغییر نداده است؛ این نتیجه دربارهٔ کل دیتاست یا نبود هرگونه رابطه تعمیم داده نمی‌شود.

نتیجهٔ holdout را هم ببینید: تعداد خوشه‌ای که فقط از بخش آموزش انتخاب می‌شود ممکن است با مدل کل داده متفاوت باشد. خوشهٔ یک‌عضوی در holdout نشانهٔ نیاز به دادهٔ بیشتر است. امتیازهای داخلی کل داده، تضمین عملکرد آینده نیستند.

For demographic segmentation, inspect the lower tail of stability, not only its median. A low ARI at the 10th percentile means some modest changes in the customer sample alter the partition substantially.

Behavior cluster descriptions (descriptive, not causal):
- Cluster 0: n=81, mean income=55.3 k$, mean spending score=49.5/100, mean age=42.7.
- Cluster 1: n=39, mean income=86.5 k$, mean spending score=82.1/100, mean age=32.7.
- Cluster 2: n=22, mean income=25.7 k$, mean spending score=79.4/100, mean age=25.3.
- Cluster 3: n=35, mean income=88.2 k$, mean spending score=17.1/100, mean age=41.1.
- Cluster 4: n=23, mean income=26.3 k$, mean spending score=20.9/100, mean age=45.2.

KMeans-only search-aware permutation diagnostic: observed=0.5547, null 95th percentile=0.4260, Monte Carlo p=0.010.
Primary model supports prediction for new rows: True.

These are the best tested candidates under the stated objective, geometry and stability penalty; they are not proof of a global optimum. Different feature scopes answer different questions. Full-data internal scores are selection scores, not unbiased estimates of future performance.

Customer-level labels: customer_segments.csv. Full search: search_results.csv. Stability: stability_trials.csv. Independent train-selected holdout: holdout_diagnostic.csv.