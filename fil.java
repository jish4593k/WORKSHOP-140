import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

public class IncomePrediction {

    public static void main(String[] args) throws Exception {
        // Load CSV data
        CSVLoader loader = new CSVLoader();
        loader.setSource(new BufferedReader(new FileReader("/content/AdultIncome.csv")));
        Instances data = loader.getData();

        // Handle missing values if any
        // (Weka automatically handles missing values during the modeling process)

        // Convert categorical variables to binary (one-hot encoding)
        // (Weka automatically handles nominal attributes during the modeling process)

        // Split data into features and target
        data.setClassIndex(data.numAttributes() - 1);

        // Split the data into training and testing sets
        Instances[] splitData = splitData(data, 80);
        Instances trainData = splitData[0];
        Instances testData = splitData[1];

        // Train Decision Tree
        J48 decisionTree = new J48();
        decisionTree.buildClassifier(trainData);

        // Evaluate Decision Tree
        Evaluation evalTree = new Evaluation(trainData);
        evalTree.evaluateModel(decisionTree, testData);
        System.out.println("Decision Tree Accuracy: " + evalTree.pctCorrect());

        // Train Random Forest
        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(trainData);

        // Evaluate Random Forest
        Evaluation evalForest = new Evaluation(trainData);
        evalForest.evaluateModel(randomForest, testData);
        System.out.println("Random Forest Accuracy: " + evalForest.pctCorrect());
    }

    // Function to split data into training and testing sets
    public static Instances[] splitData(Instances data, int percentage) {
        int trainSize = (int) Math.round(data.numInstances() * percentage / 100.0);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        return new Instances[]{train, test};
    }
}
