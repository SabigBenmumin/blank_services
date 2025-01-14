# BlankSpace Services API
## preparation environment
### create virtual environments
**install** `virtualenv` (virtual environment manager)
```bash
pip install virtualenv
```
**create** environment
- create **server** virtual environment
```bash
virtualenv serverenv
```
- create **worker** virtual environment
```bash
virtualenv workerenv
```
### how to activate virtual environment
**Windows :**
```bash
.\<environmentname>\Scripts\activate
```
**Linux :**
```bash
source <environmentname>\bin\activate
```
### install packages
**server side**
- activate `serverenv`
- install packages from [server environment requirements](requirements/serverenv_requirements.txt) file
```bash
pip install -r requirements/serverenv_requirements.txt
```
**worker side**
- activate `workerenv`
- install package what open3d require
```bash
pip install -r requirements/open3d_requirements.txt
```
- install package from [worker environment](requirements/workerenv_requirements.txt)
```bash
pip install -r requirements/workerenv_requirements.txt
```