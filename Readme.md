## Instalasi
1. Buat virtual environment:
    ```sh
    python -m venv seimbangin
    ```
    
2. Aktifkan virtual environment:
    - Windows:
        ```sh
        seimbangin\Scripts\activate
        ```
3. Instal dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Jalankan aplikasi:
    ```sh
    uvicorn seimbangin_api:app --reload
    ```
