package utility;

import org.kohsuke.args4j.Option;

public class CmdArgs
{

	@Option(name = "-model", usage = "Specify model", required = true)
	public String model = "";

	@Option(name = "-corpus", usage = "Specify path to topic modeling corpus")
	public String corpus = "";

	@Option(name = "-vectors", usage = "Specify path to the file containing word vectors")
	public String vectors = "dataset/glove.6B.200d.txt";

	@Option(name = "-nlongdoc", usage = "Specify number of pseudo-long documents")
	public int nLongDoc = 300;

	@Option(name = "-threshold", usage = "Specify threshold, unimportant correspondences between short texts and pseudo-documents are filtered")
	public double threshold = 0.001;

	@Option(name = "-GPUthreshold", usage = "Specify threshold in GPUDMM")
	public double GPUthreshold = 0.1;

	@Option(name = "-weight", usage = "Specify weight in GPUDMM")
	public double weight = 0.7;

	@Option(name = "-filterSize", usage = "Filter less semantic related word pairs in GPUDMM")
	public int filterSize = 20;

	@Option(name = "-ntopics", usage = "Specify number of topics")
	public int ntopics = 20;

	@Option(name = "-alpha", usage = "Specify alpha")
	public double alpha = 0.1;

	@Option(name = "-beta", usage = "Specify beta")
	public double beta = 0.01;

	@Option(name = "-gamma", usage = "Specify beta")
	public double gamma = 0.01;

	@Option(name = "-lambda", usage = "Specify mixture weight lambda")
	public double lambda = 0.6;

	@Option(name = "-niters", usage = "Specify number of iterations")
	public int niters = 1000;

	@Option(name = "-twords", usage = "Specify number of top topical words")
	public int twords = 20;

	@Option(name = "-maxTd", usage = "Specify number of maximum topic in each document in GPU-PDMM (use 'tau' sympol in original paper)" )
	public int maxTd = 3;

	@Option(name = "-searchTopK", usage = " for Searching Space Pruning in GPU-PDMM.")
	public int searchTopK = 10;

	@Option(name = "-window", usage = "Specify the size of windon in WNTM method")
	public int window = 10;

	@Option(name = "-name", usage = "Specify a name to topic modeling experiment")
	public String expModelName = "model";

	@Option(name = "-initFile")
	public String initTopicAssgns = "";

	@Option(name = "-sstep")
	public int savestep = 0;

	@Option(name = "-dir")
	public String dir = "";

	@Option(name = "-label")
	public String labelFile = "";

	@Option(name = "-prob")
	public String prob = "";

	@Option(name = "-topWords")
	public String topWords = "";

	@Option(name = "-paras", usage = "Specify path to hyper-parameter file")
	public String paras = "";

}
