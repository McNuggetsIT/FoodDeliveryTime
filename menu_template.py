import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

root = None  

def mostra_immagine(titolo, percorso):
        win = tk.Toplevel(root)
        win.title(titolo)

        img = Image.open(percorso)
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(win, image=img_tk)
        label.image = img_tk 
        label.pack()

def mostra_testo(titolo, testo):
    win = tk.Toplevel(root)
    win.title(titolo)

    label = tk.Label(win, text=testo, wraplength=400, justify="left")
    label.pack(padx=20, pady=20)

def main():
    
    global root
    
    root = tk.Tk()
    root.title("RISULTATI ANALISI")
    root.geometry("300x300")
    
    # Titolo
    tk.Label(root, text="RISULTATI ANALISI", font=("Arial", 14, "bold")).pack(pady=10)

    # Pulsanti menu
    tk.Button(root, text="1. Grafico di correlazione",
            command=lambda: mostra_immagine("Correlazione", "corr.png")
    ).pack(fill="x", padx=20, pady=5)

    tk.Button(root, text="2. Grafico di pairplot",
            command=lambda: mostra_immagine("Pairplot", "dist_ord.png")
    ).pack(fill="x", padx=20, pady=5)

    tk.Button(root, text="3. Grafico a barre distribuzioni",
            command=lambda: mostra_immagine("Distribuzioni", "dist.png")
    ).pack(fill="x", padx=20, pady=5)

    tk.Button(root, text="4. Analisi dati conclusioni",
            command=lambda: mostra_testo(
                "Analisi dati conclusioni",
                "Droppati tutti i valori nulli e le colonne ininfluenti per l'analisi "
                "del machine learning come: ID, precipitation. Trasformato successivamente "
                "le colonne di tipo object in valori numerici."
            )
    ).pack(fill="x", padx=20, pady=5)

    tk.Button(root, text="5. Analisi ML conclusioni",
            command=lambda: mostra_testo(
                "Analisi ML conclusioni",
                "Con il machine learning utilizzando un algoritmo di regressione "
                "abbiamo cercato di predire il tempo di consegna dei corrieri "
                "in base al luogo."
            )
    ).pack(fill="x", padx=20, pady=5)

    tk.Button(root, text="6. Esci", command=root.quit).pack(fill="x", padx=20, pady=10)

    root.mainloop()
                
if __name__ == "__main__":
    main()
else: 
    print("È stato importato")