graph TD
%% --- Style Definitions ---
classDef class fill:#f9f,stroke:#333,stroke-width:2px;
classDef main-flow fill:#cff,stroke:#333,stroke-width:2px;
classDef io fill:#fec,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
classDef lib fill:#eee,stroke:#999,stroke-width:1px;

    %% --- Main Execution Flow ---
    Start([__name__ == "__main__"]) --. Boucle sur ticker_list .-> MainFunc[(main(ticker))];
    Start --. Collecte les résumés .-> Concat[pd.concat(summaries)];
    Concat --> CSVSummary([all_signals.csv]);
    Concat --> PDFGen[Génération PDF];
    PDFGen --> LibReportlab(reportlab);
    PDFGen --> PDFSummary([summary_signals.pdf]);

    subgraph main(ticker) [main-flow]
        direction TB
        %% Instantiation
        MainFunc -->|1.| InstMS{MomentumStrategy(ticker)};

        %% Feature Engineering
        MainFunc -->|2.| MS_Features[Feature Engineering];
        MS_Features --> MS_getRSI(ms.getRSI);
        MS_Features --> MS_PriceMom(ms.PriceMomentum);
        MS_Features --> MS_getLag(ms.getLagReturns);
        MS_Features --> MS_PriceAccel(ms.PriceAccel);
        MS_Features --> MS_getPctHigh(ms.getPct52WeekHigh);
        MS_Features --> MS_getPctLow(ms.getPct52WeekLow);
        MS_Features --> MS_getVol(ms.getVol);
        MS_Features --> MS_getMacro(ms.getMacroData);
        MS_Features --> MS_getFeatures(ms.getFeaturesDataSet);

        %% Labeling & Weighting
        MainFunc -->|3.| MS_Labels(ms.getLabels);
        MainFunc -->|4.| MS_Weights(ms.getSampleWeight);

        %% Primary Model
        MainFunc -->|5.| MS_Primary(ms.PrimaryModel);

        %% Meta Features
        MainFunc -->|6.| Meta_Features[Meta-Feature Engineering];
        Meta_Features --> MS_getEntropy(ms.getEntropy);
        Meta_Features --> MS_getMaxProb(ms.getMaxProbability);
        Meta_Features --> MS_getMargin(ms.getMarginConfidence);
        Meta_Features --> MS_getF1(ms.getF1Scoredata);
        Meta_Features --> MS_getAcc(ms.getAccuracydata);

        %% Meta Model
        MainFunc -->|7.| MS_MetaLabel(ms.metaLabeling);
        MainFunc -->|8.| MS_MetaModel(ms.MetaModel);
        MainFunc -->|9.| MS_Confidence(ms.computeConfidenceScore);

        %% Bet Sizing
        MainFunc -->|10.| InstBS{BetSizing(ticker)};
        MainFunc -->|11.| BS_Calls[Appels bs.getlastPrice() & bs.position_size_with_atr()];

        %% Summary
        MainFunc -->|12.| FuncSummary[summarize_signal(ms, ...)];
    end

    %% --- Class Details: MomentumStrategy ---
    subgraph MomentumStrategy [class]
        direction TB
        InstMS --> MS_Init[__init__];
        MS_Init --> MS_getDataLoad(getDataLoad);
        MS_getDataLoad --> LibYF(yfinance);
        MS_getMacro --> LibYF;
        MS_getMacro --> LibPDR(pandas_datareader);

        MS_Weights --> InstSW{SampleWeights(...)};
        MS_Primary --> InstPKF1{PurgedKFold(...)};
        InstPKF1 --> PKF_split(split);

        MS_MetaModel --> InstPKF2{PurgedKFold(...)};
        InstPKF2 --> PKF_split;
    end
    classDef ms-class fill:#E6E6FA;
    class MomentumStrategy,MS_Init,MS_getDataLoad,MS_Features,MS_Labels,MS_Weights,MS_Primary,Meta_Features,MS_MetaLabel,MS_MetaModel,MS_Confidence,MS_getRSI,MS_PriceMom,MS_getLag,MS_PriceAccel,MS_getPctHigh,MS_getPctLow,MS_getVol,MS_getMacro,MS_getFeatures,MS_getEntropy,MS_getMaxProb,MS_getMargin,MS_getF1,MS_getAcc ms-class;

    %% --- Class Details: SampleWeights ---
    subgraph SampleWeights [class]
        direction TB
        InstSW --> SW_Init[__init__];
        InstSW -.-> SW_getInd(getIndMatrix);
        InstSW -.-> SW_getRarity(getRarity);
        InstSW -.-> SW_getRecency(getRecency);
        InstSW -.-> SW_getBootstrap(getSequentialBootstrap);
        SW_getBootstrap --> SW_getUniqueness(getAverageUniqueness);
        SW_getUniqueness --> SW_getInd;
    end
    classDef sw-class fill:#E0F2F1;
    class SampleWeights,SW_Init,SW_getInd,SW_getRarity,SW_getRecency,SW_getBootstrap,SW_getUniqueness sw-class;

    %% --- Class Details: PurgedKFold ---
    subgraph PurgedKFold [class]
        direction TB
        InstPKF1 --> PKF_Init[__init__];
        InstPKF2 --> PKF_Init;
        PKF_Init -.-> PKF_split;
    end
    classDef pkf-class fill:#FFF9C4;
    class PurgedKFold,PKF_Init,PKF_split pkf-class;

    %% --- Class Details: BetSizing ---
    subgraph BetSizing [class]
        direction TB
        InstBS --> BS_Init[__init__];
        BS_Calls --> BS_getPrice(getlastPrice);
        BS_Calls --> BS_getSize(position_size_with_atr);
        BS_getPrice --> LibYF;
    end
    classDef bs-class fill:#FBE9E7;
    class BetSizing,BS_Init,BS_getPrice,BS_getSize bs-class;

    %% --- Class Details: Backtest (Non utilisé dans le flux principal) ---
    subgraph Backtest [class]
        direction TB
        BT_Note[Non appelé dans __main__];
        InstBT{Backtest(ms)} --> BT_Init[__init__];
        BT_Init -.-> BT_Portfolio(portfolio);
        BT_Portfolio --> LibVBT(vectorbt);
    end
    classDef bt-class fill:#ECEFF1,stroke-dasharray: 5 5;
    class Backtest,BT_Note,InstBT,BT_Init,BT_Portfolio bt-class;

    %% --- Global Function ---
    FuncSummary --> SummaryData{{summary_data}};
    SummaryData --> Concat;

    %% --- Final Styling ---
    class Start,MainFunc main-flow;
    class LibReportlab,LibYF,LibPDR,LibVBT lib;
    class CSVSummary,PDFSummary io;
