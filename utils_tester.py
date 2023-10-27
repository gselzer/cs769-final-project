import utils

def main():
    wa = utils.WordAligner()
    wa.word_alignments(source_file='train.sample.en',
                       target_file='train.sample.de',
                       output_file='eflomal.en.de')

if __name__ == "__main__": main()
