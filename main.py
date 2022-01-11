import numpy as np
import tabulate as tab


class HiddenMarkovModel:
    def __init__(self):
        self.transition_list = np.zeros((45, 45))  # Macierz licząca przejść stanów - liczba_przejść_do_stanu0, liczba_przejść_do_stanu1, ..., liczba_przejść_do_stanu44}                                       #
        self.emission_list = [{} for x in range(0, 45)]  # Lista słowników z wyrażeniami dla każdego stanu - {wyrażenie0_1: liczba_wystąpień_wyrażenie1, wyrażenie0_2: liczba_wystąpień_wyrażenie2, ... , wyrażenie0_x: liczba_wystąpień_wyrażeniex},
        # {wyrażenie1_1: liczba wystąpień wyrażenie1, ..., wyrażenie1_x: liczba wystąpień_wyrażeniex}, ... {wyrażenie44_1: liczba_wystąpień_wyrażenie1, ..., wyrażenie44_x: liczba_wystąpień_wyrażeniex}
        self.tag_dict = {}  # Słownik części mowy - część_mowy: nr_indeksu

        self.transition_prob = np.zeros(
            (45, 45))  # Struktury zawierające prawdopodobieństwa przejścia stanów i emisji obserwacji.
        self.emission_prob = [{} for x in range(0, 45)]

    def initialize_tags(self, tag_lines):  # Funkcja przypisująca częściom mowy w tag_dict indeksy 0-44.
        for line in tag_lines:
            tag_data = line.split()
            self.tag_dict[tag_data[1]] = int(tag_data[0])

    def initialize_lists(self, input_lines):
        previous_state = 0
        for line in input_lines:
            if line != "\n":
                data = line.split()
                expression = data[0].lower()  # Przetwarzane wyrażenie...
                tag_index = self.tag_dict[data[1]]  # ...i odpowiadająca mu część mowy.
                if expression in self.emission_list[tag_index]:
                    self.emission_list[tag_index][
                        expression] += 1  # Jeżeli wyrażenie jest zawarte w słowniku, dodaj 1 do wartości definicji...
                else:
                    self.emission_list[tag_index][
                        expression] = 1  # ...a jeśli nie, to stwórz wpis w słowniku odpowiadający wyrażeniu, i ustaw definicję na 1.
                self.transition_list[previous_state][
                    tag_index] += 1  # Zaaktualizuj listę przejść, dodając 1 na indeksie [i][j], gdzie i to indeks stanu poprzedniego, a j to indeks stanu obecnego
                previous_state = tag_index
            else:
                previous_state = 0

    def initialize_probabilities(self):
        for index, line in enumerate(self.transition_list):
            incidence_sum = np.sum(line)  # Suma wszystkich przejść stanu.
            if incidence_sum == 0:  # Jeżeli stan nie ma przejść, bo nie występuje, to wypełnij wiersz zerami.
                new_row = [0] * 45
            else:
                new_row = np.array([value / incidence_sum for value in
                                    line])  # Lista ilorazów przejść do stanów 0,1,2,...,44 i sumy wszystkich przejść stanów.
            self.transition_prob[index] = new_row
        for index, line in enumerate(self.emission_list):
            incidence_sum = sum(
                [value for key, value in line.items()])  # Suma wszystkich obserwacji (wyrażeń) danego stanu.
            if incidence_sum == 0:
                new_row = {}  # Jeżeli stan nie ma obserwacji, bo nie występuje, to wstaw pusty słownik.
            else:
                new_row = {key: value / incidence_sum for key, value in line.items()}
            self.emission_prob[index] = new_row

    def initialize(self, input_lines, tag_lines):
        self.initialize_tags(tag_lines)
        self.initialize_lists(input_lines)
        self.initialize_probabilities()

    def visualize(self):
        print("Wizualizacja grafu przejść stanów: ")
        print(tab.tabulate(self.transition_prob, showindex="always", headers=["->s0", "->s1", "->s2", "->s3", "->s4",
                                                                              "->s5", "->s6", "->s7", "->s8", "->s9",
                                                                              "->s10", "->s11", "->s12", "->s13",
                                                                              "->s14", "->s15",
                                                                              "->s16", "->s17", "->s18", "->s19",
                                                                              "->s20", "->s21", "->s22", "->s23",
                                                                              "->s24", "->s25",
                                                                              "->s26", "->s27", "->s28", "->s29",
                                                                              "->s30", "->s31", "->s32", "->s33",
                                                                              "->s34", "->s35",
                                                                              "->s36", "->s37", "->s38", "->s39",
                                                                              "->s40", "->s41", "->s42", "->s43",
                                                                              "->s44"]))


def main():
    with open("train_input.txt") as input_file, open("taglist.txt") as taglist:
        input_lines = input_file.readlines()
        taglist_lines = taglist.readlines()
        hmm = HiddenMarkovModel()
        hmm.initialize(input_lines, taglist_lines)
        hmm.visualize()


if __name__ == '__main__':
    main()
