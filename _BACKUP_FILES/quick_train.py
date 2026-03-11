"""
Quick Training Script - Run this first to train models
"""

from advanced_train_numerical import AdvancedNumericalTrainer

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING ADVANCED NUMERICAL MODEL TRAINING")
    print("="*80 + "\n")
    
    trainer = AdvancedNumericalTrainer(
        csv_path='thyroid_cancer_risk_data.csv',
        target_accuracy=0.85
    )
    
    success = trainer.run_complete_pipeline()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("📁 Check the 'models' folder for saved models")
        print("📊 Check generated PNG files for visualizations")
        print("\n🚀 Next step: Run the web application")
        print("   python app_enhanced.py")
    else:
        print("\n❌ Training failed. Check the error messages above.")
