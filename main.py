import numpy as np

class HiddenMarkovModel:
    def __init__(self):
        self.transition_list = np.zeros((45, 45))                           # Lista przejść stanów - liczba_przejść_do_stanu0, liczba_przejść_do_stanu1, ..., liczba_przejść_do_stanu44}
        self.emission_dict = {}                                             # Słownik z wyrażeniami - wyrażenie: [liczba_wystąpień_stan0, liczba_wystąpień_stan1, ... , liczba_wystąpień_stan44]
        self.tag_dict = {}                                                  # Słownik części mowy - część_mowy: nr_indeksu

    def initialize_tags(self, tag_lines):                                   # Funkcja przypisująca częściom mowy w tag_dict indeksy 0-44.
        for line in tag_lines:
            tag_data = line.split()
            self.tag_dict[tag_data[1]] = int(tag_data[0])
    def initialize_dicts(self, input_lines):
        previous_state = 0
        for line in input_lines:
            if line != "\n":
                data = line.split()
                expression = data[0]                                        # Przetwarzane wyrażenie...
                tag_index = self.tag_dict[data[1]]                          # ...i odpowiadająca mu część mowy.
                if expression in self.emission_dict:
                    self.emission_dict[expression][tag_index] += 1          # Jeżeli wyrażenie jest zawarte w słowniku, dodaj 1 na pozycji odpowiadającej części mowy, jaką jest wyrażenie...
                else:
                    self.emission_dict[expression] = [0] * 45               # ...a jeśli nie, to stwórz wpis w słowniku odpowiadający wyrażeniu, i zainicjalizuj listę 0, poza jedną 1 odpowiadającą części mowy, jaką jest wyrażenie.
                    self.emission_dict[expression][tag_index] = 1
                self.transition_list[previous_state][tag_index] += 1        # Zaaktualizuj listę przejść, dodając 1 na indeksie [i][j], gdzie i to indeks stanu poprzedniego, a j to indeks stanu obecnego
                previous_state = tag_index
            else:
                previous_state = 0
    def print_training_data(self):
        for line in self.transition_list:
            print(line)
        for line in self.emission_dict:
            print(line)





def main():
    with open("train_input.txt") as input_file, open("taglist.txt") as taglist:
        input_lines = input_file.readlines()
        taglist_lines = taglist.readlines()
        hmm = HiddenMarkovModel()
        hmm.initialize_tags(taglist_lines)
        hmm.initialize_dicts(input_lines)
        hmm.print_training_data()

    # Use a breakpoint in the code line below to debug your script.
    # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
