"""
Quick Ensemble Training Script
Trains and combines all 4 ML models for superior performance
"""

from ensemble_train_numerical import EnsembleNumericalTrainer

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎯 STARTING ENSEMBLE TRAINING")
    print("="*80)
    print("This will train 4 base models + 3 ensemble methods:")
    print("  • Logistic Regression")
    print("  • Support Vector Machine (SVM)")
    print("  • Random Forest")
    print("  • XGBoost")
    print("  • Voting Ensemble (combines all 4)")
    print("  • Stacking Ensemble (meta-learner)")
    print("  • Weighted Ensemble (performance-based weights)")
    print("="*80 + "\n")
    
    trainer = EnsembleNumericalTrainer('thyroid_cancer_risk_data.csv')
    success = trainer.run_complete_pipeline()
    
    if success:
        print("\n✅ Ensemble training completed successfully!")
        print("📁 Check the 'models' folder for saved ensemble models")
        print("📊 Check generated PNG files for visualizations")
        print("\n🚀 Next step: Use ensemble predictions in your app")
        print("   Update app to use: utils/ensemble_predictor.py")
    else:
        print("\n❌ Ensemble training failed. Check the error messages above.")
