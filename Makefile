.PHONY: limpiar backend frontend arbol #Indica que estas son tareas que no 
# corresponden a archivos si no que son propios (locales) del proyecto.
limpiar:
	cls
	cls
	cls
	@echo "Limpiando el proyecto..."	

backend:
	@echo "Compilando el backend..."
	make limpiar
	uv run fastapi dev app/main.py --port 8000 

frontend:
	@echo "Compilando el frontend (streamlit)..."
	make limpiar
	uv run streamlit run frontend/app.py  

arbol:
	@echo "Mostrando la estructura del proyecto..."
	make limpiar
	tree /f