import utils

def test_1(source_file, target_file):
    '''
    Tests JDR method, which uses word alignment, generates a phrase translation table, and performs
    joint dropout on the source-target sentance pairs

    Output files after JDR will be 'test-data/jdr.src.en' and 'test-data/jdr.trg.en'
    '''
    wa = utils.WordAligner()
    wa.word_alignments(source_file=source_file,
                       target_file=target_file,
                       output_file='test-data/eflomal.en.de',
                       model = '3')
    JDR = utils.JointDropout(debug=False)
    JDR.joint_dropout(source_file,target_file,'test-data/eflomal.en.de', output_dir='test-data/', 
                      src_suffix='en',trg_suffix='de')
    
def main():
    test_1(source_file = 'test-data/train.sample.en.jdr',
           target_file = 'test-data/train.sample.de.jdr')

if __name__ == "__main__": main()
