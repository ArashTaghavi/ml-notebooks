# تحلیل کلاسترینگ دیتاست روت پروژه

فایل اصلی: **Clustering_Deep_Dive.ipynb**. نوت‌بوک همراه خروجی اجرا، جدول‌ها و نمودارها ذخیره شده است. دیتاست ورودی `dataset.xlsx` در روت پروژه است.

## اجرا

نوت‌بوک را در Jupyter یا VS Code باز کنید و Run All بزنید. برای اجرای اسکریپتی:

```powershell
python execute_clustering_notebook.py
```

نسخه‌های دقیق کتابخانه‌های اجرای موفق در `clustering_artifacts/requirements-lock.txt` ثبت شده‌اند. محیط اجرا Python 3.14 بوده است. نصب در محیط مجازی جداگانه توصیه می‌شود؛ نصب مجدد برای صرفاً دیدن خروجی‌ها لازم نیست.

`build_clustering_notebook.py` سازندهٔ نوت‌بوک است. اجرای آن نوت‌بوک را از نو می‌سازد و خروجی‌های داخل آن را پاک می‌کند؛ سپس باید اسکریپت اجرا را دوباره اجرا کرد.

## خروجی‌ها

- `clustering_artifacts/RESULTS.md`: نتیجهٔ انتخاب مدل، تنظیمات و تفسیر محدودیت‌ها.
- `customer_segments.csv`: دادهٔ مشتری همراه سه برچسب خوشه، سیلوئت فردی و علامت نقاط مرزی.
- `search_results.csv`: تمام ۱۴۰۷ تنظیم آزمایش‌شده، شامل هشدارها و نتایج نامعتبر.
- `validated_leaderboard.csv` و `stability_trials.csv`: رتبه‌بندی و ۷۶۰ بازبرازش برای سنجش پایداری.
- `holdout_diagnostic.csv`: بررسی تکمیلی با انتخاب تنظیمات فقط روی بخش آموزش.
- `robustness.csv`: حساسیت به seed، خطای اندازه‌گیری کوچک و حذف تک‌تک مشتری‌ها.
- `best_behavior_model.joblib`: مدل رفتار خرید همراه پیش‌پردازش آموزش‌دیده.
- `best_all_features_bundle.joblib`: مدل تمام ویژگی‌های غیرشناسه همراه پیش‌پردازش.
- `best_behavior_age_bundle.joblib`: مدل درآمد، امتیاز خرید و سن.
- `run_manifest.json`: نسخه‌ها، seed و هش دیتاست برای بازتولید.

تمام فایل‌های فهرست بالا به‌جز نوت‌بوک و اسکریپت‌ها داخل `clustering_artifacts` هستند.

## پیش‌بینی برای مشتری جدید

مدل‌های انتخاب‌شده در این اجرا KMeans هستند و از `predict` پشتیبانی می‌کنند. نام ستون‌ها و واحدها باید مانند دادهٔ پاک‌شده باشند. شمارهٔ خوشه رتبه یا نمره نیست.

```python
import joblib
import pandas as pd

bundle = joblib.load('clustering_artifacts/best_behavior_model.joblib')
new_customers = pd.DataFrame({
    'Annual Income (k$)': [70, 25],
    'Spending Score (1-100)': [80, 20],
})
labels = bundle['model'].predict(
    bundle['preprocessor'].transform(new_customers)
)
```

سیلوئت «دقت» نیست. امتیاز مدل دوبعدی و مدل همهٔ ویژگی‌ها پاسخ به دو هدف متفاوت است و نباید صرفاً از روی عدد با هم رتبه‌بندی شوند. نتیجهٔ این بررسی بهترین گزینهٔ آزمایش‌شده طبق معیارهای اعلام‌شده است، نه اثبات بهترین مدل ممکن.
