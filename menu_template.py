import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkhtmlview import HTMLLabel
import markdown

# La useremo in tutte le funzioni
root = None  

def mostra_immagine(titolo, percorso):
    win = tk.Toplevel(root)
    win.title(titolo)

    img = Image.open(percorso)
    img_tk = ImageTk.PhotoImage(img)

    label = tk.Label(win, image=img_tk)
    label.image = img_tk  # IMPORTANTE per non perdere l’immagine
    label.pack()


def mostra_testo(titolo, testo):
    win = tk.Toplevel(root)
    win.title(titolo)

    label = tk.Label(win, text=testo, wraplength=400, justify="left")
    label.pack(padx=20, pady=20)
    
def mostra_readme(titolo, percorso):
    win = tk.Toplevel()
    win.title(titolo)
    win.geometry("500x500")

    # Legge il file README.md
    with open(percorso, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Converte markdown in HTML
    html_text = markdown.markdown(md_text)

    # Frame contenitore
    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    # Aggiorna l'area scrollabile
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    # Quando la finestra cambia dimensione → aggiorna larghezza del canvas
    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfig(window_id, width=e.width)
    )

    # Crea finestra nel canvas
    window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # HTMLLabel che si adatta alla larghezza del canvas
    label = HTMLLabel(scrollable_frame, html=html_text)
    label.pack(fill="both", expand=True, padx=10, pady=10)


def main():
    global root   # ci serve per usarlo nelle altre funzioni

    root = tk.Tk()
    root.title("RISULTATI ANALISI")
    root.geometry("500x500")

    tk.Label(root, text="RISULTATI ANALISI", font=("Arial", 14, "bold")).pack(pady=10)

    tk.Button(root, text="1. Grafico di correlazione",
              command=lambda: mostra_immagine("Correlazione", "corr.png")
    ).pack(fill="x", padx=20, pady=5)

    tk.Button(root, text="2. Distribuzione ordine",
              command=lambda: mostra_immagine("Distribusione ordine", "dist_ord.png")
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
    
    tk.Button(root, text="6. Crediti",
          command=lambda: mostra_readme("Crediti", "README.md")
    ).pack(fill="x", padx=20, pady=5)

    tk.Button(root, text="7. Esci", command=root.quit).pack(fill="x", padx=20, pady=10)

    root.mainloop()
                
if __name__ == "__main__":
    main()
else: 
    print("È stato importato")