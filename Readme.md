## Instalasi
1. Buat virtual environment:
    ```sh
    python -m venv seimbangin
    ```
Aktifkan virtual environment:
    - Windows:
        ```sh
        seimbangin\Scripts\activate
        ```
Instal dependencies:
    ```sh
    pip install -r requirements.txt
    ```
Jalankan aplikasi:
    ```sh
    uvicorn seimbangin_api:app --reload
    ```