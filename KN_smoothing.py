'''
El algoritmo de Kneser-Ney Smoothing es una técnica para manejar el problema de palabras desconocidas y suavizar las probabilidades.

Para esta implementación de Kneser-Ney interpolado se van a manejar los siguientes aspectos:

    1. Manejar un corpus grande
    2. Optimizar el cálculo de frecuencias y probabilidades
    3. Extensión a trigramas y n-diagramas superiores
    4. Modularidad para orden y endendimiento.

Otros conceptos importantes:

n-grama

    Una secuencia de 'n' elementos (pueden ser palabras, caracteres, fonemas, etc.) que se extraen de un texto o un corpus de texto. 
    Tipos de n-gramas:

        Unigrama (n=1): Una secuencia de un solo elemento. En "el gato come pescado", Unigramas (n=1): ["el", "gato", "come", "pescado"]
        Bigramas (n=2): Un bigrama es una secuencia de dos elementos consecutivos. En "el gato come pescado", Bigramas (n=2): ["el gato", "gato come", "come pescado"]
        Trigramas (n=3): Un trigrama es una secuencia de tres elementos consecutivos. Trigramas (n=3): ["el gato come", "gato come pescado"]
        n-gramas (n=k): Una secuencia de 'n' elementos consecutivos. El valor de 'n' puede ser cualquier número entero positivo

'''

from collections import defaultdict, Counter
import math

class KneserNeySmoothing:
    def __init__(self, corpus, n=2, descuento=0.75):
        """
        Inicializar el modelo Kneser-Ney Smoothing:

            a) corpus (list of str): Una lista de oraciones que representa el corpus
            b) n (int): El orden de los n-gramas
            c) descuento (float): El valor de descuento D utilizado en Kneser-Ney

        """
        self.n = n
        self.descuento = descuento
        self.unigramCounts = Counter()
        self.ngramCounts = defaultdict(Counter)
        self.cuentasContinuacion = Counter()
        self.cuentasContexto = Counter()
        self.totalNgrams = 0
        self.totalUnigrams = 0
        
        # Calcular las frecuencias de unigramas y n-gramas
        self._calcularFrecuencias(corpus)
        # Calcular las cuentas de continuación
        self._calcularCuentasContinuacion()
        # Calcular las cuentas de contexto
        self._calcularCuentasContexto()

    def _calcularFrecuencias(self, corpus):
        """
        Calcular las frecuencias de unigramas y n-gramas en el corpus:

            a) corpus (list of str): Una lista de oraciones que representa el corpus

        """
        for oracion in corpus:
            # Dividir la oración en tokens (palabras) usando el espacio como delimitador
            tokens = oracion.split()
            # Incrementar el total de unigramas por la cantidad de tokens en la oración
            self.totalUnigrams += len(tokens)
            # Actualizar el contador de unigramas con los tokens de la oración
            self.unigramCounts.update(tokens)
            # Crear una lista de n-gramas a partir de los tokens
            ngrams = zip(*[tokens[i:] for i in range(self.n)])
            # Iterar sobre cada n-grama y actualiza el contador de n-gramas
            for ngram in ngrams:
                # Aumentar la cuenta del n-grama específico
                self.ngramCounts[len(ngram)][ngram] += 1
                # Incrementar el total de n-gramas
                self.totalNgrams += 1
    
    def _calcularCuentasContinuacion(self):
        """
        Calcula las cuentas de continuación, que representan cuántas veces un
        n-grama aparece como continuación de cualquier contexto

        """
        # Iterar sobre cada n-grama en los conteos de n-gramas de longitud n
        for ngram in self.ngramCounts[self.n]:
            # Incrementar la cuenta de continuación para el sufijo del n-grama (excluyendo la primera palabra)
            self.cuentasContinuacion[ngram[1:]] += 1
    
    def _calcularCuentasContexto(self):
        """
        Calcula las cuentas de contexto, que representan la frecuencia de los
        contextos (los prefijos de longitud n-1)

        """
        # Iterar sobre cada n-grama en los conteos de n-gramas de longitud n
        for ngram in self.ngramCounts[self.n]:
            # Extraer el prefijo del n-grama (excluyendo la última palabra)
            context = ngram[:-1]
            # Incrementar la cuenta del contexto
            self.cuentasContexto[context] += 1
    
    def _probabilidadContinuacion(self, word):
        """
        Calcula la probabilidad de continuación de una palabra:

            a) word (str): La palabra para la cual calcular la probabilidad de continuación

        Retorna:

            a) float: La probabilidad de continuación de la palabra

        """
        # Calcular la probabilidad de continuación dividiendo las cuentas de continuación de la palabra por el total de n-gramas
        return self.cuentasContinuacion[(word,)] / self.totalNgrams

    def _probabilidadCondicional(self, ngram):
        """
        Calcula la probabilidad condicional suavizada usando Kneser-Ney:

            a) ngram (tuple of str): El n-grama para el cual calcular la probabilidad

        Retorna:
            
            a) float: La probabilidad condicional del n-grama

        """
        # Caso base: si el n-grama es un unigrama, se devuelve la probabilidad
        # de la palabra unigram tomando su cuenta y dividiéndola por el total de unigramas
        if len(ngram) == 1:
            return self.unigramCounts[ngram[0]] / self.totalUnigrams
        
        # Dividir el n-grama en su prefijo (todas las palabras excepto la última) y la última palabra
        prefix, word = ngram[:-1], ngram[-1]
        
        # Obtener la cuenta del prefijo en los conteos de contexto
        prefixCount = self.cuentasContexto[prefix]
        
        # Si el prefijo no tiene ninguna ocurrencia, usar la probabilidad de continuación de la palabra
        if prefixCount == 0:
            return self._probabilidadContinuacion(word)
        
        # Obtener la cuenta del n-grama específico en los conteos de n-gramas
        ngramCount = self.ngramCounts[len(ngram)][ngram]
        
        # Calcular la probabilidad de orden inferior recursivamente
        lowerOrderProb = self._probabilidadCondicional(ngram[1:])
        
        # Calcular y devuelve la probabilidad condicional suavizada usando la fórmula de Kneser-Ney
        return max(ngramCount - self.descuento, 0) / prefixCount + self.descuento * lowerOrderProb / prefixCount

    def obtenerProbabilidad(self, ngram):
        """
        Devuelve la probabilidad de un n-grama específico:

            a) ngram (tuple of str): El n-grama para el cual calcular la probabilidad

        Retorna:
        
            a) float: La probabilidad del n-grama

        """
        return self._probabilidadCondicional(ngram)

    def generarProbabilidadOracion(self, oracion):
        """
        Calcula la probabilidad de una oración completa:

            a) oracion (str): La oración para la cual calcular la probabilidad

        Retorna:
        
            a) float: La probabilidad de la oración

        """
        # Dividir la oración en tokens (palabras) usando el espacio como delimitador
        tokens = oracion.split()
        
        # Generar una lista de n-gramas a partir de los tokens.
        # zip(*[tokens[i:] for i in range(self.n)]) crea una lista de tuplas
        # donde cada tupla contiene 'n' elementos consecutivos de la lista tokens
        ngrams = zip(*[tokens[i:] for i in range(self.n)])
        
        # Inicializar la probabilidad total de la oración como 1.0
        prob = 1.0
        
        # Itera rsobre cada n-grama y multiplica la probabilidad total por la probabilidad del n-grama
        for ngram in ngrams:
            prob *= self.obtenerProbabilidad(ngram)
        
        # Devolver la probabilidad total de la oración
        return prob

#------------------------------------------------------------------------------------------------------------------

# Corpus, oraciones
corpus = [
    "el gato come pescado",
    "el perro come carne",
    "el gato duerme",
    "el perro ladra",
    "el perro come pescado"
]

# Crear una instancia del modelo
kneserNey = KneserNeySmoothing(corpus, n=2, descuento=0.75)

# Obtener las probabilidades de algunos bigramas
print("P('gato'|'el'):", kneserNey.obtenerProbabilidad(('el', 'gato')))
print("P('come'|'gato'):", kneserNey.obtenerProbabilidad(('gato', 'come')))
print("P('carne'|'come'):", kneserNey.obtenerProbabilidad(('come', 'carne')))
print("P('pescado'|'perro'):", kneserNey.obtenerProbabilidad(('perro', 'pescado')))  # Ejemplo con un bigrama que no está en el corpus

# Calcular la probabilidad de una oración completa
print("Probabilidad de la oración 'el perro come carne':", kneserNey.generarProbabilidadOracion("el perro come carne"))

#------------------------------------------------------------------------------------------------------------------

'''
Interpretación de los resultados:

    P('gato'|'el'): 0.6666666666666666:
        Dado el contexto "el", la probabilidad de que la siguiente palabra sea "gato" es 0.6666666666666666

    P('come'|'gato'): 0.1875:
        Dado el contexto "gato", la probabilidad de que la siguiente palabra sea "come" es 0.1875

    P('carne'|'come'): 0.14583333333333334:
        Dado el contexto "come", la probabilidad de que la siguiente palabra sea "carne" es 0.14583333333333334

    P('pescado'|'perro'): 0.041666666666666664:
        Dado el contexto "perro", la probabilidad de que la siguiente palabra sea "pescado" es 0.041666666666666664

    Probabilidad de la oración 'el perro come carne': 0.11905924479166667:
        Esta probabilidad es el producto de las probabilidades de los bigramas que componen la oración
        Indica qué tan probable es que esta oración ocurra en el corpus según el modelo

'''