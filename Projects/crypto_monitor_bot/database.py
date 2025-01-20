from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

Base = declarative_base()

# Таблица пользователей
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, unique=True, nullable=False)
    thresholds = Column(String)  # Хранение порогов как строка, например, "50,30,10"
    dropped_coins = relationship("DroppedCoin", back_populates="user")

# Таблица упавших монет
class DroppedCoin(Base):
    __tablename__ = 'dropped_coins'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    coin = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)
    user = relationship("User", back_populates="dropped_coins")

# Инициализация базы данных
engine = create_engine('sqlite:///users.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def init_db():
    Base.metadata.create_all(engine)

def add_user(chat_id):
    existing_user = session.query(User).filter_by(chat_id=chat_id).first()
    if not existing_user:
        new_user = User(chat_id=chat_id, thresholds='')
        session.add(new_user)
        session.commit()

def update_user_thresholds(chat_id, threshold):
    user = session.query(User).filter_by(chat_id=chat_id).first()
    if user:
        current_thresholds = set(map(float, user.thresholds.split(','))) if user.thresholds else set()
        if threshold in current_thresholds:
            current_thresholds.remove(threshold)
        else:
            current_thresholds.add(threshold)
        user.thresholds = ','.join(map(str, sorted(current_thresholds, reverse=True)))
        session.commit()

def get_user_thresholds(chat_id):
    user = session.query(User).filter_by(chat_id=chat_id).first()
    if user and user.thresholds:
        return sorted(map(float, user.thresholds.split(',')), reverse=True)
    return []

def get_all_users():
    users = session.query(User).all()
    user_list = []
    for user in users:
        thresholds = list(map(float, user.thresholds.split(','))) if user.thresholds else []
        user_list.append({
            'chat_id': user.chat_id,
            'thresholds': thresholds
        })
    return user_list

def add_dropped_coin(chat_id, coin, threshold):
    user = session.query(User).filter_by(chat_id=chat_id).first()
    if user:
        # Проверяем, есть ли уже такая монета с таким порогом
        existing = session.query(DroppedCoin).filter_by(user_id=user.id, coin=coin, threshold=threshold).first()
        if not existing:
            new_dropped = DroppedCoin(user_id=user.id, coin=coin, threshold=threshold)
            session.add(new_dropped)
            session.commit()

def remove_dropped_coin(chat_id, coin):
    user = session.query(User).filter_by(chat_id=chat_id).first()
    if user:
        dropped = session.query(DroppedCoin).filter_by(user_id=user.id, coin=coin).all()
        for d in dropped:
            session.delete(d)
        session.commit()

def get_dropped_coins(chat_id):
    user = session.query(User).filter_by(chat_id=chat_id).first()
    if user:
        return session.query(DroppedCoin).filter_by(user_id=user.id).all()
    return []
