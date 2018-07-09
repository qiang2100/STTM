import models.*;


import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import utility.CmdArgs;
import eval.ClusteringEval;

/**
 * STTM: A Java package for the short text topic models including DMM, BTM, SATM, LDA, LFDMM, LFLDA, etc.
 *
 * 
 * @author: Jipeng Qiang
 * 
 * @version: 1.0
 * 
 */
public class STTM
{
	public static void main(String[] args)
	{

		CmdArgs cmdArgs = new CmdArgs();
		CmdLineParser parser = new CmdLineParser(cmdArgs);
		try {

			parser.parseArgument(args);

			if (cmdArgs.model.equals("LDA")) {
				LDA lda = new LDA(cmdArgs.corpus,
					cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
					cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
					cmdArgs.initTopicAssgns, cmdArgs.savestep);
				lda.inference();
			}
			else if (cmdArgs.model.equals("DMM")) {
				DMM dmm = new DMM(cmdArgs.corpus,
					cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
					cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
					cmdArgs.initTopicAssgns, cmdArgs.savestep);
				dmm.inference();
			}
			else if (cmdArgs.model.equals("BTM")) {
				BTM btm = new BTM(cmdArgs.corpus,
						cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				btm.inference();
			}
			else if (cmdArgs.model.equals("SATM")) {
				SATM satm = new SATM(cmdArgs.corpus,
						cmdArgs.ntopics, cmdArgs.nLongDoc, cmdArgs.threshold, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				satm.inference();
			}else if (cmdArgs.model.equals("GPUDMM")) {
				GPUDMM gpudmm = new GPUDMM(cmdArgs.corpus, cmdArgs.vectors,cmdArgs.GPUthreshold,cmdArgs.weight,cmdArgs.filterSize,
						cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				gpudmm.inference();
			}
			else if (cmdArgs.model.equals("LDAinf")) {
				LDA_Inf lda = new LDA_Inf(
					cmdArgs.paras, cmdArgs.corpus, cmdArgs.niters,
					cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
				lda.inference();
			}
			else if (cmdArgs.model.equals("DMMinf")) {
				DMM_Inf dmm = new DMM_Inf(
					cmdArgs.paras, cmdArgs.corpus, cmdArgs.niters,
					cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
				dmm.inference();
			}
			else if (cmdArgs.model.equals("LFLDA")) {
				LFLDA lflda = new LFLDA(cmdArgs.corpus, cmdArgs.vectors,
						cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.lambda, cmdArgs.niters, cmdArgs.niters,
						cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				lflda.inference();
			}
			else if (cmdArgs.model.equals("LFDMM")) {
				LFDMM lfdmm = new LFDMM(cmdArgs.corpus, cmdArgs.vectors,
						cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.lambda, cmdArgs.niters, cmdArgs.niters,
						cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				lfdmm.inference();
			}
			else if (cmdArgs.model.equals("LFLDAinf")) {
				LFLDA_Inf lfldaInf = new LFLDA_Inf(cmdArgs.paras,
						cmdArgs.corpus, cmdArgs.niters, cmdArgs.niters,
						cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
				lfldaInf.inference();
			}
			else if (cmdArgs.model.equals("LFDMMinf")) {
				LFDMM_Inf lfdmmInf = new LFDMM_Inf(cmdArgs.paras,
						cmdArgs.corpus, cmdArgs.niters, cmdArgs.niters,
						cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
				lfdmmInf.inference();
			}
			else if (cmdArgs.model.equals("Eval")) {
				ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir,
					cmdArgs.prob);
			}
			else {
				System.out
					.println("Error: Option \"-model\" must get \"LDA\" or \"DMM\" or \"BTM\" or \"SATM\" or \"GPUDMM\" or \"LDAinf\" or \"DMMinf\" or \"Eval\"");
				System.out
					.println("\tLDA: Specify the Latent Dirichlet Allocation topic model");
				System.out
					.println("\tDMM: Specify the one-topic-per-document Dirichlet Multinomial Mixture model");
				System.out
						.println("\tBTM: Infer topics for Biterm");
				System.out
						.println("\tSATM: Infer topics using SATM");
				System.out
						.println("\tGPUDMM: Infer topics using GPUDMM");
				System.out
					.println("\tLDAinf: Infer topics for unseen corpus using a pre-trained LDA model");
				System.out
					.println("\tDMMinf: Infer topics for unseen corpus using a pre-trained DMM model");
				System.out
					.println("\tEval: Specify the document clustering evaluation");
				help(parser);
				return;
			}
		}
		catch (CmdLineException cle) {
			System.out.println("Error: " + cle.getMessage());
			help(parser);
			return;
		}
		catch (Exception e) {
			System.out.println("Error: " + e.getMessage());
			e.printStackTrace();
			return;
		}

		System.out.println("end!!!!!!!");
	}

	public static void help(CmdLineParser parser)
	{
		System.out
			.println("java -jar jLDADMM.jar [options ...] [arguments...]");
		parser.printUsage(System.out);
	}
}
