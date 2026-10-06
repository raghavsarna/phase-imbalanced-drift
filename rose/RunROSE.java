import java.io.PrintWriter;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.meta.imbalanced.ROSE;
import moa.streams.ArffFileStream;

/**
 * Runs ROSE (Cano & Krawczyk, 2022) with default options on an ARFF stream
 * and writes one predicted class index per scored sample.
 *
 *   java RunROSE <stream.arff> <n_train> <seed> <out.txt>
 *
 * n_train > 0 : temporal hold-out -- learn the first n_train samples, then
 *               predict the remaining ones without updating the model;
 * n_train = 0 : prequential -- predict every sample, then learn from it.
 */
public class RunROSE {
    private static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    public static void main(String[] args) throws Exception {
        String arff = args[0];
        int nTrain = Integer.parseInt(args[1]);
        int seed = Integer.parseInt(args[2]);
        ArffFileStream stream = new ArffFileStream(arff, -1);
        stream.prepareForUse();
        ROSE rose = new ROSE();
        rose.setRandomSeed(seed);
        rose.setModelContext(stream.getHeader());
        rose.prepareForUse();
        try (PrintWriter out = new PrintWriter(args[3])) {
            int t = 0;
            while (stream.hasMoreInstances()) {
                Instance x = stream.nextInstance().getData();
                if (t == 0) rose.getVotesForInstance(x);      // ROSE allocates its ensemble on the first query
                if (nTrain > 0) {
                    if (t < nTrain) rose.trainOnInstance(x);
                    else out.println(argmax(rose.getVotesForInstance(x)));
                } else {
                    out.println(argmax(rose.getVotesForInstance(x)));
                    rose.trainOnInstance(x);
                }
                t++;
            }
        }
    }
}
