# README.md

## Environment Setup

Make sure you have the following versions installed:

- **Conda Version:** 23.7.4
- **Python Version:** 3.11.5

## Package Installation

```bash
pip install pandas==2.1.4
pip install numpy==1.26.3
pip install requests==2.31.0
pip install finance-datareader==0.9.66
pip install beautifulsoup4==4.12.2
pip install plotly==5.18.0
pip install logging==0.4.9.6
pip install scikit-learn==1.3.2
pip install tensorflow==2.15.0

pip install Django==5.0.1
pip install mysqlclient=2.2.1
```

## Django Setup
```bash
python -m venv Django
Django\Scripts\activate.bat

django-admin startproject stockviewer

python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## Database Operations
```sql
-- Creating Database
CREATE DATABASE stock_prediction;
USE stock_prediction;

-- Dropping Tables (if exist)
DROP TABLE IF EXISTS STOCK_PREDICTION;
DROP TABLE IF EXISTS STOCK_DATA;
DROP TABLE IF EXISTS STOCK_LIST;
DROP TABLE IF EXISTS MACHINE;

-- Creating Tables
CREATE TABLE MACHINE (
    name varchar(10),
    machine_binary binary,
    is_used bit,
    create_date datetime,
    update_date datetime,
    PRIMARY KEY (name)
);

CREATE TABLE STOCK_LIST (
    code varchar(10),
    name varchar(200),
    is_used bit,
    create_date datetime,
    update_date datetime,
    PRIMARY KEY (code)
);

CREATE TABLE STOCK_DATA (
    code varchar(10),
    date datetime,
    Open DECIMAL(16,2), 
    High DECIMAL(16,2),
    Low DECIMAL(16,2),
    Close DECIMAL(16,2),
    `Change` DECIMAL(16,2),
    Volume DECIMAL(16,2),
    5MvAvg DECIMAL(16,2),
    20MvAvg DECIMAL(16,2),
    60MvAvg DECIMAL(16,2),
    112MvAvg DECIMAL(16,2),
    224MvAvg DECIMAL(16,2),
    UpperBand DECIMAL(16,2),
    LowerBand DECIMAL(16,2),
    20VolAvg DECIMAL(16,2),
    60VolAvg DECIMAL(16,2),
    TranAmnt DECIMAL(16,2),
    is_used bit,
    create_date datetime,
    update_date datetime,
    PRIMARY KEY (code, date),
    FOREIGN KEY (code) REFERENCES STOCK_LIST(code)
);

CREATE TABLE STOCK_PREDICTION (
    code varchar(10),
    date datetime,
    machine varchar(10),
    Open DECIMAL(16,2), 
    High DECIMAL(16,2),
    Low DECIMAL(16,2),
    Close DECIMAL(16,2),
    Volume DECIMAL(16,2),
    is_used bit,
    create_date datetime,
    update_date datetime,
    PRIMARY KEY (code, date, machine),
    FOREIGN KEY (code) REFERENCES STOCK_LIST(code),
    FOREIGN KEY (machine) REFERENCES MACHINE(name)
);
```