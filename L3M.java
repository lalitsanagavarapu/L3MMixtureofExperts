import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.Logistic;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class L3M {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}

	public static Evaluation classify(Classifier model, Instances trainingSet,
			Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
		System.out
				.println("\n L3M Classification Results\n =================================");
		System.out.println("\n Correctly Classified Instances : "
				+ evaluation.correct());
		System.out.println("\n InCorrectly Classified Instances : "
				+ evaluation.incorrect());
		System.out.println("\n Standard Deviation : "
				+ evaluation.rootMeanSquaredError());
		// Metrics
		System.out.println(evaluation.toSummaryString());
		//
		System.out.println(evaluation.toMatrixString());
		return evaluation;
	}

	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
		return 100 * correct / predictions.size();
	}

	public static Instances[][] crossValidationSplit(Instances data,
			int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
		return split;
	}

	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("C://Program Files//Weka-3-6//data//image-segmentation.arff");
		Instances data = new Instances(datafile);
		//
		int iter = 1; // computation time calculation !
		for (int i = 0; i < iter; i++) {
			System.out.println("Start System Time for our L3M is "
					+ System.currentTimeMillis());
			Remove remove = new Remove();
			remove.setInvertSelection(true);
			remove.setInputFormat(data);
			//
			AddCluster filter = new AddCluster();
			String[] options1 = new String[2];
			options1[0] = "-W"; // "cluster options"
			options1[1] = "weka.clusterers.EM -I 10 -N 6 -M 1.0E-6 -S 100";
			filter.setOptions(options1);
			filter.setInputFormat(data);
			data = Filter.useFilter(data, filter);
			// System.out.println("Filtered data:" + data);
			//
			data.setClassIndex(data.numAttributes() - 1);
			// Do 10-split cross validation
			// Instances[][] split = crossValidationSplit(data, 10);
			// Separate split into training and testing arrays
			// Instances[] trainingSplits = split[0];
			// Instances[] testingSplits = split[1];
			//
			Logistic Lg = new Logistic();
			String[] options = { "-R 1.0E-8 -M 200" };
			Lg.setOptions(options);
			//
			int trainSize = (int) Math.round(data.numInstances() * 0.090909091);
			int testSize = data.numInstances() - trainSize;
			//

			Instances train = new Instances(data, 0, trainSize);
			Instances test = new Instances(data, trainSize, testSize);
			//
			// train = Filter.useFilter(data, remove);
			//
			FastVector predictions = new FastVector();
			// for (int i = 0; i < trainingSplits.length; i++) {
			//
			// Evaluation validation = classify(Lg, trainingSplits[i],
			// testingSplits[i]);
			Evaluation validation = classify(Lg, train, test);
			predictions.appendElements(validation.predictions());
			// }
			// Calculate overall accuracy of current classifier on all splits
			double accuracy = calculateAccuracy(predictions);
			//
			ThresholdCurve tc = new ThresholdCurve();
			ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
			vmc.setROCString("(Area under ROC = "
					+ Utils.doubleToString(tc.getROCArea(data), 4) + ")");
			vmc.setName(train.relationName());
			PlotData2D tempd = new PlotData2D(data);
			tempd.setPlotName(data.relationName());
			tempd.addInstanceNumberAttribute();
			// specify which points are connected
			boolean[] cp = new boolean[data.numInstances()];
			for (int n = 1; n < cp.length; n++)
				cp[n] = true;
			// tempd.setConnectPoints(cp);
			// add plot
			vmc.addPlot(tempd);
			// method visualizeClassifierErrors
			String plotName = vmc.getName();
			final javax.swing.JFrame jf = new javax.swing.JFrame(
					"SMAI Classifier Graph: " + plotName);
			jf.setSize(500, 500);
			// jf.getContentPane().setLayout(new BorderLayout());
			jf.getContentPane().add(vmc, BorderLayout.CENTER);
			jf.addWindowListener(new java.awt.event.WindowAdapter() {
				public void windowClosing(java.awt.event.WindowEvent e) {
					jf.dispose();
				}
			});
			jf.setVisible(true);
			System.out.println("End System Time for our L3M is "
					+ System.currentTimeMillis());
			System.out
					.println("\n Overall Accuracy of our L3M Classification : "
							+ String.format("%.2f%%", accuracy)
							+ "\n----------------------------------------------------------");
			// }
		}
	}
}
