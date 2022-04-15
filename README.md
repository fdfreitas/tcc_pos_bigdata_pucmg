# MT

Fãbio Daros de Freitas
Repositório do Trabalho de Conclusão de Curso Ciência de Dados - Turma 2020 (PUC-MG)

* [1. Introduction](#introduction)
* [2. Files and Installation](#files-and-installation)
* [3. About string encoding](#about-string-encoding)
* [4. Notes on context creation](#notes-on-context-creation)
* [5. Usage](#usage)
* [6. TODO](#todo)
* [7. Changes](#changes)
* [8. Donation](#donation)

## Introduction


## Files and Installation

Dependencies:
- [poetry](https://python-poetry.org/docs/basic-usage/)
- [mql-zmq](https://github.com/dingmaotu/mql-zmq)
- [MQL5-JSON-API](https://github.com/khramkov/MQL5-JSON-API)
- [backtrader](https://github.com/khramkov/Backtrader-MQL5-API)
- Metatrader 5 installation tree at "MT\MetaTrader 5 Terminal" dir<br>
This is not included (.gitignore'd) in repo

### When creating project MT
    mkdir MT
    cd MT
    
#### Dependencies installing
    ./py_env_config.sh
    poetry install
    poetry shell
    mkdir -p SHARE/PKGS
    cd SHARE/PKGS
    git clone https://github.com/dingmaotu/mql-zmq
    rm -rf mql-zmq/.git*
    git clone https://github.com/khramkov/MQL5-JSON-API
    rm -rf MQL5-JSON-API/.git*
    git clone https://github.com/khramkov/Backtrader-MQL5-API
    git clone https://github.com/Oxylo/btreport.git
    rm -rf btreport/.git
    git commit -am 'Added submodules'
    rm -rf Backtrader-MQL5-API/.git*
    cd Backtrader-MQL5-API
    sudo apt install python3-pip
    pip3 install -r requirements.txt
    pip3 install .
    sudo aptitude install python3-tk
    cd ..
    cd ..
    cd ..
    ./mql-zmq_install.sh
    ./MQL5-JSON-API_install.sh
    ? pip install quantstats --upgrade --no-cache-dir

### When cloning project MT
    git clone https://ita.no-ip.org:MT.git
    cd MT
    
#### Dependencies installing
    ./py_env_config.sh
    poetry install
    poetry shell
    pip install -r requirements.txt
    cd SHARE/PKGS
    cd Backtrader-MQL5-API
    sudo apt install python3-pip
    pip3 install -r requirements.txt
    pip3 install .
    sudo aptitude install python3-tk
    cd ..
<!--    cd btreports
    pip install -r requirements.txt-->
    cd ..
    cd ..
    cd ..
    ./mql-zmq_install.sh
    ./MQL5-JSON-API_install.sh

    
## Preparing for run
- If poetry is not in path, include ~/.poetry/bin/ 
- Meta Trader preparation for MQL5-JSON-API:
  - Compile MT\MetaTrader 5 Terminal\MQL5\Experts\JsonAPI.mq5 script.
  - Check if Metatrader 5 automatic trading is allowed.
  - Attach the JsonAPI.mq5 script to a chart in Metatrader 5.
  - Allow DLL import in dialog window.
  - Check if the ports are free to use. (default:15555,15556,15557,15558,15559,15560,15562)
  
- poetry shell
- run strategies under SRC directory
  
## TODO

1. Scripting for live strategies running.

## Changes
