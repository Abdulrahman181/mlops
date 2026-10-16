# Engineering Review — `vortextech-aiml-internship`

## النطاق

تم تدقيق المستودع المحلي `Abdulrahman181/vortextech-aiml-internship` فقط، بما في ذلك:

- `week1.ipynb` و`week2.ipynb`.
- `README.md` و`requirements.txt` و`.gitignore`.
- سلامة JSON وصياغة Python داخل خلايا Jupyter، والمخرجات المحفوظة.
- قابلية نقل مسار البيانات، تقسيم البيانات، تسرب المعالجة، ثبات النتائج، والمقاييس.

لم يتم دفع أي تغيير إلى GitHub. كانت هناك تغييرات محلية سابقة على الفرع قبل هذه المراجعة؛ أُبقيت كما هي وأُضيفت إصلاحات هذا التقرير في commit محلي جديد فقط.

## النتائج الحرجة والقيود

1. **البيانات غير موجودة في المستودع أو بيئة Sandbox.** لا يوجد `train.csv` متعقب أو محلي، لذلك لم يُنفذ تشغيل كامل للدفاتر ولم تُخترع نتائج أو مقاييس. أصبح كل دفتر يتوقف برسالة `FileNotFoundError` واضحة تطلب ضبط `TITANIC_TRAIN_CSV`.
2. **اعتماديات التشغيل الثقيلة غير مثبتة في البيئة الحالية.** `scikit-learn` و`jupyter` غير متاحين، لذلك اقتصرت المصادقة على الفحص الثابت. تتوافر بعض مكتبات التحليل (`numpy`, `pandas`, `matplotlib`, `seaborn`) لكن ذلك لا يكفي لتشغيل `week2.ipynb`.
3. قبل الإصلاح، كان `week2.ipynb` يعالج القيم المفقودة ويحوّل المتغيرات الفئوية قبل التقسيم؛ وهذا يسمح بتسرب إحصاءات مجموعة الاختبار إلى التدريب. كما كان التقسيم غير stratified.
4. قبل الإصلاح، لم يكن تحميل البيانات يتحقق من وجود الملف أو مخطط الأعمدة، وكان مسار الخطأ الافتراضي غير موثق بما يكفي للتشغيل من بيئة مختلفة.
5. كانت الاعتماديات غير محددة الحدود ولم تتضمن Jupyter رغم أن README يطلب تشغيله.

## الإصلاحات المنفذة

### الدفاتر

- `week1.ipynb`:
  - استخدام `pathlib.Path` و`TITANIC_TRAIN_CSV` مع دعم `~`.
  - رسائل خطأ مفهومة للملف المفقود، الملف الفارغ، والأعمدة المطلوبة الناقصة.
  - إزالة الخلايا الرسومية المكررة، وتنظيف التكرارات بعد التحميل.
  - إبقاء المعالجة موصوفة بوضوح على أنها للتحليل الوصفي فقط.
- `week2.ipynb`:
  - التحقق من الملف والمخطط والهدف الثنائي والحد الأدنى لتكرار الفئات.
  - إزالة التكرارات قبل التقسيم لتقليل احتمال وجود السجل نفسه في المجموعتين.
  - `train_test_split(..., stratify=y, random_state=42)`.
  - وضع `SimpleImputer` و`OneHotEncoder(handle_unknown="ignore")` داخل `ColumnTransformer` و`Pipeline` لا يتعلم إلا من بيانات التدريب.
  - استبعاد `PassengerId` والحقول الحرة `Name` و`Ticket` و`Cabin` من خط الأساس البسيط.
  - تثبيت seed للنموذج، واستخدام `zero_division=0` في F1.
  - لم تُحفظ أي مخرجات تنفيذ قديمة في ملفات notebooks.

### ملفات المشروع

- `scripts/validate_static.py`: أداة stdlib قابلة للتشغيل تفحص AST لكل ملفات Python، وJSON/AST لخلايا notebooks، وتمنع execution counts أو outputs المحفوظة.
- `requirements.txt`: إضافة حدود إصدارات معقولة لـ NumPy/Pandas/Matplotlib/Seaborn/scikit-learn وإضافة Jupyter.
- `README.md`: توثيق مصدر البيانات، الأمر الموصى به للمسار المطلق، أمر التحقق، فصل المعالجة عن الاختبار، وحدود عدم توفر البيانات.
- `.gitignore`: إضافة ملفات CSV والمخلفات الشائعة لـ Ruff/Pytest والصور/PDF المحلية.

## التحقق المنفذ

| الفحص | النتيجة |
|---|---|
| `python3 scripts/validate_static.py` | نجح: Python syntax وNotebook JSON/code وغياب outputs/execution counts |
| `ruff check --isolated --select E9,F63,F7,F82 scripts /tmp/vortex-ruff` | نجح: لا syntax errors أو undefined names في الأداة وخلايا الدفاتر المستخرجة |
| `git diff --check` | نجح: لا whitespace errors |
| فحص الملفات والبيانات المتعقبة | لا توجد بيانات `csv` أو `data/` في المستودع |
| فحص الاعتماديات دون تثبيت | `numpy`, `pandas`, `matplotlib`, `seaborn` متاحة؛ `scikit-learn` و`jupyter` غير متاحين |

لم يتم ادعاء تشغيل كامل أو عرض أرقام Accuracy/F1، لأن ملف البيانات و`scikit-learn` غير متوفرين في البيئة.

## طريقة التشغيل بعد توفير البيانات

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export TITANIC_TRAIN_CSV=/absolute/path/to/train.csv
python scripts/validate_static.py
jupyter notebook
```

شغّل خلايا كل دفتر من الأعلى إلى الأسفل، وراجع مخطط الأعمدة وتوازن `Survived` قبل تفسير أي نتيجة.

## Second Senior Review (2026-10-16)

A second manual pass rechecked both internship notebooks, Python syntax, notebook execution state, and static validation. The repository lacked regression tests, so `tests/test_validate_static.py` was added to exercise the validator and enforce output-free, execution-count-free notebooks. No dataset-dependent results were claimed.

Validation: `python3 scripts/validate_static.py`, `pytest -q` (2 passed), `compileall`, and `git diff --check` passed.
