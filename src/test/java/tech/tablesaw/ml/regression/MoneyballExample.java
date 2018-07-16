/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tech.tablesaw.ml.regression;

import com.sun.deploy.trace.Trace;
import tech.tablesaw.api.*;
import tech.tablesaw.api.plot.Histogram;
import tech.tablesaw.api.plot.Scatter;
import tech.tablesaw.api.ml.regression.LeastSquares;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;
import tech.tablesaw.table.TableSliceGroup;

import java.util.List;

/**
 * An example doing ordinary least squares regression
 */
public class MoneyballExample {

    public static void main(String[] args) throws Exception {

        // Get the data
        Table baseball = Table.read().csv("data/baseball.csv");
        out(baseball.structure());

        // filter to the data available in the 2002 season
        Table moneyball = baseball.where(QueryHelper.numberColumn("year").isLessThan(2002));

        // plot regular season wins against year, segregating on whether the team made the plays
        NumberColumn wins = moneyball.numberColumn("W");
        NumberColumn year = moneyball.numberColumn("Year");
        CategoricalColumn playoffs = moneyball.categoricalColumn("Playoffs");
        scatterPlot("Regular season wins by year", moneyball,"W", "Year", "playoffs");
        //Scatter.show("Regular season wins by year", wins, year, moneyball.splitOn(playoffs));

        // Calculate the run difference for use in the regression model
        NumberColumn runDifference = moneyball.numberColumn("RS").subtract(moneyball.numberColumn("RA"));
        moneyball.addColumns(runDifference);
        runDifference.setName("RD");

        // Plot RD vs Wins to see if the relationship looks linear
        scatterPlot("RD x Wins", moneyball, "RD","W");
        //Scatter.show("RD x Wins", moneyball.numberColumn("RD"), moneyball.numberColumn("W"));

        // Create the regression model
        //NumberColumn wins = moneyball.numberColumn("W");
        LeastSquares winsModel = LeastSquares.train(wins, runDifference);
        out(winsModel);

        // Make a prediction of how many games we win if we score 135 more runs than our opponents
        double[] testValue = new double[1];
        testValue[0] = 135;
        double prediction = winsModel.predict(testValue);
        out("Predicted wins with RD = 135: " + prediction);

        // Predict runsScored based on On-base percentage, batting average and slugging percentage

        LeastSquares runsScored = LeastSquares.train(moneyball.nCol("RS"),
                moneyball.nCol("OBP"), moneyball.nCol("BA"), moneyball.nCol("SLG"));
        out(runsScored);

        LeastSquares runsScored2 = LeastSquares.train(moneyball.nCol("RS"),
                moneyball.nCol("OBP"), moneyball.nCol("SLG"));
        out(runsScored2);

        Histogram.show(runsScored2.residuals());

        //Scatter.show(runsScored2.fitted(), runsScored2.residuals());

        scatterPlot("Fitted vs Residuals", "Fitted", runsScored2.fitted(), "Residuals", runsScored2.residuals());

        //Scatter.show(runsScored2.actuals(), runsScored2.residuals());
        scatterPlot("Actuals vs Residuals", "Actuals", runsScored2.actuals(), "Residuals", runsScored2.residuals());

        // We use opponent OBP and opponent SLG to model the efficacy of our pitching and defence

        Table moneyball2 = moneyball.where(QueryHelper.numberColumn("year").isGreaterThan(1998));
        LeastSquares runsAllowed = LeastSquares.train(moneyball2.nCol("RA"),
                moneyball2.nCol("OOBP"), moneyball2.nCol("OSLG"));
        out(runsAllowed);
    }

    private static void out(Object o) {
        System.out.println(String.valueOf(o));
    }

    private static void scatterPlot(String title, Table table, String xCol, String yCol, String groupCol) {

        TableSliceGroup tables = table.splitOn(table.categoricalColumn(groupCol));

        Layout layout = Layout.builder()
                .title(title)
                .height(600)
                .width(800)
                .xAxis(Axis.builder()
                        .title(xCol)
                        .build())
                .yAxis(Axis.builder()
                        .title(yCol)
                        .build())
                .build();

        ScatterTrace[] traces  = new ScatterTrace[2];
        for (int i = 0; i < tables.size(); i++) {
            List<Table> tableList = tables.asTableList();
            traces[i] = ScatterTrace.builder(
                    tableList.get(i).numberColumn(xCol),
                    tableList.get(i).numberColumn(yCol))
                    .build();
        }
        Figure figure = new Figure(layout, traces);
        Plot.show(figure);
    }

    private static void scatterPlot(String title, Table table, String xCol, String yCol) {

        Layout layout = Layout.builder()
                .title(title)
                .height(600)
                .width(800)
                .xAxis(Axis.builder()
                        .title(xCol)
                        .build())
                .yAxis(Axis.builder()
                        .title(yCol)
                        .build())
                .build();

        ScatterTrace trace = ScatterTrace.builder(
                    table.numberColumn(xCol),
                    table.numberColumn(yCol))
                    .build();
        Figure figure = new Figure(layout, trace);
        Plot.show(figure);
    }

    private static void scatterPlot(String title, String xTitle, double[] xCol, String yTitle, double[] yCol) {

        Layout layout = Layout.builder()
                .title(title)
                .height(600)
                .width(800)
                .xAxis(Axis.builder()
                        .title(xTitle)
                        .build())
                .yAxis(Axis.builder()
                        .title(yTitle)
                        .build())
                .build();

        ScatterTrace trace = ScatterTrace.builder(xCol, yCol).build();
        Figure figure = new Figure(layout, trace);
        Plot.show(figure);
    }
}
