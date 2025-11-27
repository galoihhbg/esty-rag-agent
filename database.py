"""
Database Module - Quản lý kết nối và operations với PostgreSQL/SQLite
Supports both PostgreSQL (production) and SQLite (testing/development)
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class TrainingExample(Base):
    """Model cho training examples"""
    __tablename__ = 'training_examples'
    
    id = Column(Integer, primary_key=True)
    user_input = Column(Text, nullable=False)
    correct_output = Column(JSON, nullable=False)
    category = Column(String(100), default='general')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_validated = Column(Boolean, default=False)


class ConfigField(Base):
    """Model cho config fields"""
    __tablename__ = 'config_fields'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    field_type = Column(String(50), nullable=False)
    options = Column(JSON)
    is_required = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Color(Base):
    """Model cho colors"""
    __tablename__ = 'colors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    hex_code = Column(String(7))
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Model cho prediction logs"""
    __tablename__ = 'prediction_logs'
    
    id = Column(Integer, primary_key=True)
    user_input = Column(Text, nullable=False)
    config_used = Column(JSON)
    color_list = Column(JSON)  # Store as JSON for cross-database compatibility
    result = Column(JSON)
    used_examples = Column(JSON)  # Store as JSON for cross-database compatibility
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Class quản lý database operations"""
    
    def __init__(self, database_url: Optional[str] = None, test_mode: bool = False):
        """
        Khởi tạo DatabaseManager.
        
        Args:
            database_url: URL kết nối database
            test_mode: Nếu True, sử dụng SQLite in-memory
        """
        self.test_mode = test_mode
        
        if test_mode:
            # Test mode: sử dụng SQLite in-memory
            self.engine = create_engine(
                'sqlite:///:memory:',
                connect_args={'check_same_thread': False},
                poolclass=StaticPool
            )
        else:
            db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///./data/etsy_rag.db')
            self.engine = create_engine(db_url)
        
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Tạo database session"""
        return self.SessionLocal()
    
    # Training Examples Operations
    def add_training_example(self, user_input: str, correct_output: Any, category: str = 'general') -> int:
        """Thêm training example mới"""
        with self.get_session() as session:
            example = TrainingExample(
                user_input=user_input,
                correct_output=correct_output,
                category=category
            )
            session.add(example)
            session.commit()
            return example.id
    
    def get_training_examples(self, category: Optional[str] = None, validated_only: bool = False) -> List[Dict]:
        """Lấy danh sách training examples"""
        with self.get_session() as session:
            query = session.query(TrainingExample)
            if category:
                query = query.filter(TrainingExample.category == category)
            if validated_only:
                query = query.filter(TrainingExample.is_validated == True)
            
            return [
                {
                    'id': ex.id,
                    'user_input': ex.user_input,
                    'correct_output': ex.correct_output,
                    'category': ex.category,
                    'is_validated': ex.is_validated,
                    'created_at': ex.created_at.isoformat() if ex.created_at else None
                }
                for ex in query.all()
            ]
    
    def validate_training_example(self, example_id: int, is_validated: bool = True) -> bool:
        """Đánh dấu training example đã được validate"""
        with self.get_session() as session:
            example = session.query(TrainingExample).filter(TrainingExample.id == example_id).first()
            if example:
                example.is_validated = is_validated
                example.updated_at = datetime.utcnow()
                session.commit()
                return True
            return False
    
    def delete_training_example(self, example_id: int) -> bool:
        """Xóa training example"""
        with self.get_session() as session:
            example = session.query(TrainingExample).filter(TrainingExample.id == example_id).first()
            if example:
                session.delete(example)
                session.commit()
                return True
            return False
    
    # Config Fields Operations
    def add_config_field(self, name: str, field_type: str, options: Optional[Dict] = None, is_required: bool = True) -> int:
        """Thêm config field mới"""
        with self.get_session() as session:
            field = ConfigField(
                name=name,
                field_type=field_type,
                options=options,
                is_required=is_required
            )
            session.add(field)
            session.commit()
            return field.id
    
    def get_config_fields(self) -> List[Dict]:
        """Lấy danh sách config fields"""
        with self.get_session() as session:
            return [
                {
                    'id': f.id,
                    'name': f.name,
                    'type': f.field_type,
                    'options': f.options,
                    'is_required': f.is_required
                }
                for f in session.query(ConfigField).all()
            ]
    
    def delete_config_field(self, field_id: int) -> bool:
        """Xóa config field"""
        with self.get_session() as session:
            field = session.query(ConfigField).filter(ConfigField.id == field_id).first()
            if field:
                session.delete(field)
                session.commit()
                return True
            return False
    
    # Colors Operations
    def add_color(self, name: str, hex_code: Optional[str] = None) -> int:
        """Thêm color mới"""
        with self.get_session() as session:
            color = Color(name=name, hex_code=hex_code)
            session.add(color)
            session.commit()
            return color.id
    
    def get_colors(self) -> List[Dict]:
        """Lấy danh sách colors"""
        with self.get_session() as session:
            return [
                {
                    'id': c.id,
                    'name': c.name,
                    'hex_code': c.hex_code
                }
                for c in session.query(Color).all()
            ]
    
    def delete_color(self, color_id: int) -> bool:
        """Xóa color"""
        with self.get_session() as session:
            color = session.query(Color).filter(Color.id == color_id).first()
            if color:
                session.delete(color)
                session.commit()
                return True
            return False
    
    # Prediction Logs Operations
    def log_prediction(self, user_input: str, config_used: Any, color_list: List[str], 
                       result: Any, used_examples: List[str]) -> int:
        """Lưu prediction log"""
        with self.get_session() as session:
            log = PredictionLog(
                user_input=user_input,
                config_used=config_used,
                color_list=color_list,
                result=result,
                used_examples=used_examples
            )
            session.add(log)
            session.commit()
            return log.id
    
    def get_prediction_logs(self, limit: int = 100) -> List[Dict]:
        """Lấy prediction logs"""
        with self.get_session() as session:
            logs = session.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit).all()
            return [
                {
                    'id': log.id,
                    'user_input': log.user_input,
                    'config_used': log.config_used,
                    'color_list': log.color_list,
                    'result': log.result,
                    'used_examples': log.used_examples,
                    'created_at': log.created_at.isoformat() if log.created_at else None
                }
                for log in logs
            ]
    
    # Validation Statistics
    def get_validation_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về dữ liệu validation"""
        with self.get_session() as session:
            total_examples = session.query(TrainingExample).count()
            validated_examples = session.query(TrainingExample).filter(TrainingExample.is_validated == True).count()
            total_fields = session.query(ConfigField).count()
            total_colors = session.query(Color).count()
            total_predictions = session.query(PredictionLog).count()
            
            # Category breakdown
            categories = session.query(
                TrainingExample.category,
            ).distinct().all()
            
            category_stats = {}
            for (cat,) in categories:
                cat_total = session.query(TrainingExample).filter(TrainingExample.category == cat).count()
                cat_validated = session.query(TrainingExample).filter(
                    TrainingExample.category == cat,
                    TrainingExample.is_validated == True
                ).count()
                category_stats[cat] = {
                    'total': cat_total,
                    'validated': cat_validated,
                    'unvalidated': cat_total - cat_validated
                }
            
            return {
                'training_examples': {
                    'total': total_examples,
                    'validated': validated_examples,
                    'unvalidated': total_examples - validated_examples,
                    'validation_rate': round(validated_examples / total_examples * 100, 2) if total_examples > 0 else 0
                },
                'config_fields': total_fields,
                'colors': total_colors,
                'prediction_logs': total_predictions,
                'categories': category_stats,
                'is_data_sufficient': total_examples >= 10 and validated_examples >= 5 and total_fields > 0 and total_colors > 0
            }
