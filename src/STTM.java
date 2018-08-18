import eval.CoherenceEval;
import models.*;


import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import utility.CmdArgs;
import eval.ClusteringEval;
import eval.ClassificationEval;

/**
 * STTM: A Java package for the short text topic models including DMM, BTM, WNTM, PTM, SATM, LDA, LFDMM, LFLDA, etc.
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
					cmdArgs.ntopics, 50.0/cmdArgs.ntopics, cmdArgs.beta,
					cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName);
				lda.inference();
			}
			else if (cmdArgs.model.equals("DMM")) {
				DMM dmm = new DMM(cmdArgs.corpus,
					cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
					cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName);
				dmm.inference();
			}
			else if (cmdArgs.model.equals("BTM")) {
				BTM btm = new BTM(cmdArgs.corpus,
						cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				btm.inference();
			}
			else if (cmdArgs.model.equals("WNTM")) {
				WNTM wntm = new WNTM(cmdArgs.corpus,
						cmdArgs.ntopics, 50.0/cmdArgs.ntopics, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.window, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				wntm.inference();
			}
			else if (cmdArgs.model.equals("SATM")) {
				SATM satm = new SATM(cmdArgs.corpus,
						cmdArgs.ntopics, cmdArgs.nLongDoc, cmdArgs.threshold, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				satm.inference();
			}else if (cmdArgs.model.equals("PTM")) {
				PTM ptm = new PTM(cmdArgs.corpus, cmdArgs.ntopics, cmdArgs.nLongDoc, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				ptm.inference();
			}
			else if (cmdArgs.model.equals("GPUDMM")) {
				GPUDMM gpudmm = new GPUDMM(cmdArgs.corpus, cmdArgs.vectors,cmdArgs.GPUthreshold,cmdArgs.weight,cmdArgs.filterSize,
						cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				gpudmm.inference();
			}else if (cmdArgs.model.equals("GPU_PDMM")) {
				GPU_PDMM gpupdmm = new GPU_PDMM(cmdArgs.corpus, cmdArgs.vectors,cmdArgs.GPUthreshold,cmdArgs.weight,cmdArgs.filterSize,
						cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda,
						cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
						cmdArgs.initTopicAssgns, cmdArgs.savestep);
				gpupdmm.inference();
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
			else if (cmdArgs.model.equals("ClusteringEval")) {
				ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir,
					cmdArgs.prob);
			}else if (cmdArgs.model.equals("ClassificationEval")) {
				ClassificationEval.evaluate(cmdArgs.labelFile, cmdArgs.dir,
						cmdArgs.prob);
			}else if (cmdArgs.model.equals("CoherenceEval")) {
					CoherenceEval.evaluate(cmdArgs.labelFile, cmdArgs.dir,
							cmdArgs.topWords);
				}
			else {
				System.out
					.println("Error: Option \"-model\" must get \"LDA\" or \"DMM\" or \"BTM\" or \"WNTM\" or \"SATM\" or \"GPUDMM\" or \"GPU_PDMM\" or \"LDALDA\" or \"LFDMM\" or \"Eval\"");
				System.out
					.println("\tLDA: Specify the Latent Dirichlet Allocation topic model");
				System.out
					.println("\tDMM: Specify the one-topic-per-document Dirichlet Multinomial Mixture model");
				System.out
						.println("\tBTM: Infer topics for Biterm");
				System.out
						.println("\tWNTM: Infer topics for WNTM");
				System.out
						.println("\tSATM: Infer topics using SATM");
				System.out
						.println("\tPTM: Infer topics using PTM");
				System.out
						.println("\tGPUDMM: Infer topics using GPUDMM");
				System.out
						.println("\tGPU_PDMM: Infer topics using GPU_PDMM");
				System.out
						.println("\tLFLDA: Infer topics using LFLDA");
				System.out
						.println("\tLFDMM: Infer topics using LFDMM");
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
			.println("java -jar jSTTM.jar [options ...] [arguments...]");
		parser.printUsage(System.out);
	}
}
