# Përdorimi i Teknikave të Machine Learning për Zbulimin e Mashtrimeve në Transaksionet Financiare

## Abstrakti

Ky punim trajton përdorimin e teknikave të Machine Learning për identifikimin e transaksioneve mashtruese në pagesat financiare elektronike. Studimi është realizuar mbi dataset-in `creditcard.csv`, i cili përmban transaksione me kartë krediti dhe një variabël objektiv `Class`, ku vlera `1` përfaqëson mashtrim dhe vlera `0` përfaqëson transaksion normal. Në këtë punim janë krahasuar tre modele: `Linear Regression`, `Random Forest` dhe `XGBoost`.

Për shkak se dataset-i është shumë i pabalancuar, analiza përshkruese është bërë mbi të gjithë dataset-in, ndërsa procesi i modelimit është realizuar mbi një mostër të stratifikuar që përfshin të gjitha rastet mashtruese dhe 40,000 raste jo mashtruese. Rezultatet tregojnë se `Random Forest` dha performancën më të mirë sipas `F1-Score`, ndërsa `XGBoost` rezultoi modeli me `ROC-AUC` dhe `PR-AUC` më të lartë. Kjo tregon se algoritmet e bazuara në pemë janë më të përshtatshme për këtë lloj problemi sesa një model i thjeshtë linear.

## Fjalë kyçe

Machine Learning, mashtrime financiare, klasifikim, Random Forest, XGBoost, Linear Regression, fraud detection

## 1. Hyrja

Rritja e përdorimit të pagesave elektronike dhe sistemeve dixhitale financiare ka sjellë edhe rritje të tentativave për mashtrim. Zbulimi i shpejtë i këtyre rasteve është një nga sfidat më të rëndësishme për bankat, kompanitë e pagesave dhe institucionet financiare. Nëse një sistem nuk identifikon në kohë një transaksion të dyshimtë, humbja financiare mund të jetë e konsiderueshme.

Qasjet tradicionale të bazuara vetëm në rregulla fikse nuk janë gjithmonë të mjaftueshme, sepse mashtrimet ndryshojnë vazhdimisht. Për këtë arsye, teknikat e Machine Learning janë bërë shumë të rëndësishme, pasi ato mësojnë modele nga të dhënat historike dhe ndihmojnë në identifikimin e sjelljeve anormale.

Ky punim synon të ndërtojë një krahasim të thjeshtë dhe praktik të tre modeleve të ndryshme për zbulimin e mashtrimeve financiare, pa hyrë në optimizime shumë të avancuara.

## 2. Qëllimi dhe objektivat

Qëllimi kryesor i këtij punimi është të vlerësohet se sa mirë mund të përdoren algoritmet e Machine Learning për të klasifikuar transaksionet si normale ose mashtruese.

Objektivat kryesore janë:

1. Të analizohet dataset-i dhe shpërndarja e klasave.
2. Të trajnohen tre modele: `Linear Regression`, `Random Forest` dhe `XGBoost`.
3. Të krahasohen modelet duke përdorur metrika të rëndësishme për probleme të pabalancuara.
4. Të gjenerohen figura dhe matrica që ilustrojnë rezultatet.
5. Të identifikohet modeli më i përshtatshëm për këtë problem.

## 3. Përshkrimi i dataset-it

Dataset-i i përdorur në këtë studim është `creditcard.csv`. Ai përmban:

- `284,807` rreshta
- `31` kolona
- `492` raste mashtrimi
- `284,315` raste jo mashtrimi
- pa vlera mungese

Kolona e fundit `Class` tregon nëse transaksioni është mashtrim ose jo. Shumica e atributeve `V1` deri `V28` janë të transformuara, ndërsa `Time` dhe `Amount` përfaqësojnë kohën dhe shumën e transaksionit.

Një problem i dukshëm është pabalancimi i fortë i klasave. Rastet mashtruese përbëjnë vetëm `0.1727%` të dataset-it të plotë. Kjo do të thotë se metrika si `accuracy` nuk mjafton për të vlerësuar performancën e modeleve.

Për efikasitet llogaritës në fazën e modelimit, është përdorur një mostër e stratifikuar me:

- `40,492` transaksione gjithsej
- `492` raste mashtrimi
- `40,000` raste jo mashtrimi

Kjo qasje ruan të gjitha rastet mashtruese dhe e bën trajnimin më praktik për një projekt të thjeshtë akademik.

## 4. Metodologjia

Procesi i punës është ndarë në disa hapa:

### 4.1 Analiza përshkruese

Fillimisht u analizua dataset-i për të kuptuar shpërndarjen e klasave, shumën e transaksioneve dhe lidhjen mes disa atributeve kryesore. Për këtë janë gjeneruar:

- grafik i shpërndarjes së klasave
- histogram i shumës së transaksioneve
- scatter plot për atributet `V14` dhe `V17`
- correlation matrix për atributet më të lidhura me klasën
- boxplot i shumës së transaksioneve sipas klasës

### 4.2 Ndarja e të dhënave

Të dhënat e përdorura për modelim u ndanë në:

- `80%` për trajnim
- `20%` për testim

Ndarja u bë në mënyrë të stratifikuar për të ruajtur raportin ndërmjet klasave.

### 4.3 Modelet e përdorura

Në këtë punim u përdorën tre modele:

#### a. Linear Regression

Edhe pse `Linear Regression` përdoret zakonisht për probleme regresioni, në këtë punim u përfshi si model krahasues i thjeshtë. Daljet e modelit u kufizuan në intervalin `[0, 1]` dhe më pas u përdor pragu `0.5` për klasifikim.

#### b. Random Forest

`Random Forest` është një model i bazuar në shumë pemë vendimmarrjeje. Ai është i përshtatshëm për të kapur marrëdhënie jo-lineare dhe zakonisht performon mirë në probleme klasifikimi.

#### c. XGBoost

`XGBoost` është një algoritëm boosting shumë i përdorur në probleme praktike të klasifikimit. Ai njihet për performancë të lartë dhe për aftësinë për të trajtuar marrëdhënie komplekse mes atributeve.

### 4.4 Metrikat e vlerësimit

Për shkak të pabalancimit të klasave, u përdorën këto metrika:

- `Accuracy`
- `Precision`
- `Recall`
- `F1-Score`
- `ROC-AUC`
- `PR-AUC`

Në probleme të fraud detection, `Recall` është shumë e rëndësishme sepse tregon sa raste mashtrimi arrin të kapë modeli. `Precision` është po ashtu e rëndësishme sepse mat sa nga alarmet e ngritura janë vërtet mashtrime. `F1-Score` jep një balancë mes këtyre dy treguesve.

## 5. Rezultatet eksperimentale

Rezultatet përfundimtare të modeleve janë paraqitur më poshtë:

| Modeli | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9972 | 0.9310 | 0.8265 | 0.8757 | 0.9802 | 0.9035 |
| XGBoost | 0.9954 | 0.7748 | 0.8776 | 0.8230 | 0.9855 | 0.9138 |
| Linear Regression | 0.9947 | 0.9661 | 0.5816 | 0.7261 | 0.9795 | 0.8547 |

### 5.1 Interpretimi i rezultateve

`Random Forest` dha rezultatin më të mirë sipas `F1-Score = 0.8757`, që e bën modelin më të balancuar ndërmjet `precision` dhe `recall`. Ky model arriti gjithashtu `precision = 0.9310`, që do të thotë se shumica e rasteve të sinjalizuara si mashtrim janë vërtet mashtrime.

`XGBoost` arriti `recall = 0.8776`, që është më i larti ndër tre modelet, si dhe `ROC-AUC = 0.9855` dhe `PR-AUC = 0.9138`, që janë po ashtu vlerat më të larta. Kjo tregon se `XGBoost` është shumë i mirë në kapjen e rasteve mashtruese, edhe pse prodhon më shumë alarme të rreme sesa `Random Forest`.

`Linear Regression` pati `precision` shumë të lartë (`0.9661`), por `recall = 0.5816`, që do të thotë se humbi një pjesë të konsiderueshme të rasteve mashtruese. Për këtë arsye, ai nuk është zgjidhja më e mirë për një sistem real të zbulimit të mashtrimeve.

### 5.2 Analiza sipas raporteve të klasifikimit

Në setin e testimit kishte `98` raste mashtrimi. Nga këto:

- `Linear Regression` identifikoi rreth `58` raste
- `Random Forest` identifikoi rreth `81` raste
- `XGBoost` identifikoi rreth `86` raste

Kjo e bën të qartë se modelet me bazë pemësh janë më efektive në zbulimin e sjelljeve të dyshimta.

## 6. Vizualizimet e krijuara

Të gjitha figurat janë ruajtur në folderin `images/`. Figurave kryesore u përkasin:

- `class_distribution.png`
- `amount_distribution.png`
- `scatter_v14_v17.png`
- `correlation_matrix_top_features.png`
- `amount_boxplot_by_class.png`
- `confusion_matrix_linear_regression.png`
- `confusion_matrix_random_forest.png`
- `confusion_matrix_xgboost.png`
- `roc_curves.png`
- `model_metric_comparison.png`

Këto vizualizime ndihmojnë në interpretimin më të qartë të dataset-it dhe të sjelljes së modeleve.

## 7. Diskutimi

Rezultatet tregojnë se problemi i zbulimit të mashtrimeve financiare nuk duhet vlerësuar vetëm me `accuracy`, sepse të tre modelet kanë saktësi shumë të lartë. Kjo ndodh sepse klasa jo mashtruese dominon fort në dataset.

Nga ana praktike:

- nëse prioritet është kapja e sa më shumë rasteve mashtrimi, `XGBoost` është zgjedhje shumë e mirë
- nëse kërkohet një balancë më e mirë mes kapjes së mashtrimeve dhe reduktimit të alarmeve të rreme, `Random Forest` është më i përshtatshëm
- `Linear Regression` mund të përdoret vetëm si krahasim bazë dhe jo si zgjidhje optimale

## 8. Përfundimi

Ky punim tregoi se teknikat e Machine Learning mund të përdoren me sukses për zbulimin e mashtrimeve në transaksionet financiare. Nga tre modelet e testuara, `Random Forest` rezultoi modeli më i balancuar sipas `F1-Score`, ndërsa `XGBoost` tregoi aftësi shumë të mira në kapjen e rasteve mashtruese, duke arritur `Recall`, `ROC-AUC` dhe `PR-AUC` më të lartë.

Në përgjithësi, mund të thuhet se modelet e bazuara në pemë janë më të përshtatshme për këtë problem sesa një model linear. Kjo ndodh sepse ato arrijnë të modelojnë më mirë marrëdhëniet komplekse dhe jo-lineare në të dhëna.

## 9. Kufizimet dhe puna e ardhshme

Ky punim është mbajtur qëllimisht i thjeshtë. Disa kufizime janë:

1. Nuk është bërë optimizim i thelluar i hiperparametrave.
2. Nuk janë përdorur teknika si `SMOTE` ose `undersampling` i avancuar.
3. Nuk është bërë validim `cross-validation`.
4. `Linear Regression` nuk është model ideal për klasifikim binar, por është përdorur për krahasim sipas kërkesës.

Në të ardhmen, studimi mund të zgjerohet me:

1. `Logistic Regression` në vend të `Linear Regression`
2. `SMOTE` për balancim të klasave
3. `Grid Search` ose `Random Search` për optimizim parametrash
4. analiza të rëndësisë së veçorive

## 10. Struktura e skedarëve të projektit

- `scripts/fraud_analysis.py` për trajnim, vlerësim dhe gjenerim figurash
- `images/` për të gjitha figurat
- `results/model_metrics.csv` për tabelën e metrikave
- `results/classification_reports.txt` për raportet e klasifikimit
- `results/dataset_summary.json` për përmbledhjen e dataset-it

## 11. Udhëzim i shkurtër për ekzekutim

Për të rigjeneruar rezultatet mjafton komanda:

```bash
python3 scripts/fraud_analysis.py
```

Pas ekzekutimit, figurat ruhen automatikisht në `images/`, ndërsa metrikat ruhen në `results/`.
