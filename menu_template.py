import tkinter as tk
from PIL import Image, ImageTk

def main():
          
    while True:
        
        print("\n RISULTATI ANALISI")
        print("1. Grafico di correlazione")
        print("2. Grafico di pairplot")
        print("3. Grafico a barre distribuzioni")
        print("4. Analisi dati conclusioni")
        print("5. Analisi ml conclusioni")
        print("6. Esci")
        chooice = int(input("Scegli una opzione: "))
        
        match chooice:
            case 1:
                root = tk.Tk()
                root.title("Correlazione")

                img = Image.open("corr.png")
                img_tk = ImageTk.PhotoImage(img)

                label = tk.Label(root, image=img_tk)
                label.pack()

                root.mainloop()
            case 2:
                root = tk.Tk()
                root.title("Distanza tipo ordine")

                img = Image.open("dist_ord.png")
                img_tk = ImageTk.PhotoImage(img)

                label = tk.Label(root, image=img_tk)
                label.pack()

                root.mainloop()
            case 3:
                root = tk.Tk()
                root.title("Distribuzioni")

                img = Image.open("dist.png")
                img_tk = ImageTk.PhotoImage(img)

                label = tk.Label(root, image=img_tk)
                label.pack()

                root.mainloop()
            case 4:
                
                root = tk.Tk()
                root.title("Analisi dati conclusioni")

                testo = ("Droppati tutti i valori nulli e le colonne ininfluenti per l'analisi del machine learning come: ID, precipitation. Trasformato successivamente le colonne di tipo object in valori numerici.")

                label = tk.Label(root, text=testo, wraplength=400, justify="left")
                label.pack(padx=20, pady=20)

                root.mainloop()
            case 5:
                root = tk.Tk()
                root.title("Analisi ml conclusioni")

                testo = ("Con il machine learning utilizzando un algoritmo di regresssione "
                        "abbiamo cercato di predirre il tempo di consegna futuro dei vari "
                        "corrieri in base al luogo.")

                label = tk.Label(root, text=testo, wraplength=400, justify="left")
                label.pack(padx=20, pady=20)

                root.mainloop()
            case 6:
                break
            case _:
                print("Scelta non valida")
                
if __name__ == "__main__":
    main()
else: 
    print("È stato importato")