# BlankSpace Services API
## **prepare the environment**
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
## **run project**
This project requires running two separate terminals for the server and worker components.
> [!IMPORTANT]
> Ensure that the `uploads` directory exists in your project. If it does not, you must create the directory.
```bash
mkdir uploads
```

### Worker Setup
1. Activate the worker environment:
   ```bash
   source workerenv/bin/activate   # For Unix/Linux/MacOS
   # OR
   ./workerenv\Scripts\activate      # For Windows
2. Run the worker script:
```bash
python worker.py
```
### Server Setup
1. Activate the server environment:
   ```bash
   source serverenv/bin/activate   # For Unix/Linux/MacOS
   # OR
   ./serverenv\Scripts\activate      # For Windows
2. Run te server script
```bash
uvicorn server:app --reload
```