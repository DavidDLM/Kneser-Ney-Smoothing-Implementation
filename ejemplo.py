from collections import defaultdict, Counter

class KneserNeyLM:
    def __init__(self, n, discount=0.75):
        """
        Inicializa el modelo de lenguaje Kneser-Ney

        param n: Tamaño de los n-gramas (por ejemplo, 2 para bigramas).
        param discount: Factor de descuento, alrededor de 0.75
        """
        self.n = n
        self.discount = discount
        self.ngram_counts = defaultdict(Counter)  # Contador de n-gramas
        self.context_counts = defaultdict(int)    # Contador de contextos (n-1-gramas)
        self.vocab = set()                        # Vocabulario del corpus

    def train(self, corpus):
        """
        Entrena el modelo con el corpus dado

        :param corpus: Lista de listas de palabras (cada lista es una oración)
        """
        for sentence in corpus:
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']  # Añade marcadores de inicio y fin de oración
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
                self.vocab.add(word)

    def get_ngram_prob(self, ngram):
        """
        Calcula la probabilidad de un n-grama dado

        param ngram: Tupla de palabras que representa el n-grama
        return: Probabilidad del n-grama.
        """
        context = ngram[:-1]
        word = ngram[-1]

        # Si el contexto no está en los conteos, devuelve una probabilidad uniforme
        if context not in self.context_counts:
            return 1 / len(self.vocab)
        
        # Si el n-grama se ha visto antes, aplica el descuento
        if self.ngram_counts[context][word] > 0:
            numerator = self.ngram_counts[context][word] - self.discount
            denominator = self.context_counts[context]
            prob = numerator / denominator
        else:
            prob = 0

        # Suma la probabilidad de backoff
        return prob + self.discount * len(self.ngram_counts[context]) / self.context_counts[context] * self.get_context_prob(context, word)

    def get_context_prob(self, context, word):
        """
        Calcula la probabilidad de una palabra en un contexto dado usando backoff.

        param context: Tupla de palabras que representa el contexto
        param word: Palabra cuya probabilidad se calcula
        return: Probabilidad de la palabra en el contexto
        """
        if context in self.context_counts:
            return self.ngram_counts[context].get(word, 0) / self.context_counts[context]
        else:
            return 1 / len(self.vocab)

# Ejemplo de uso
def main():
    corpus = [
        ["this", "is", "a", "test"],
        ["this", "is", "another", "test"]
    ]

    # Creamos un modelo de bigramas (n=2)
    model = KneserNeyLM(n=2)

    # Entrenamos el modelo con el corpus
    model.train(corpus)

    # Calculamos la probabilidad del bigrama ("this", "is")
    prob = model.get_ngram_prob(("this", "is"))
    print(f"Probabilidad del bigrama ('this', 'is'): {prob}")

    # Calculamos la probabilidad del bigrama ("is", "a")
    prob = model.get_ngram_prob(("is", "a"))
    print(f"Probabilidad del bigrama ('is', 'a'): {prob}")

if __name__ == "__main__":
    main()
