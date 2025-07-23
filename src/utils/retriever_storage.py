import os
import pickle
import logging
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục lưu trữ retrievers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETRIEVER_FOLDER = os.path.join(BASE_DIR, 'src', 'assets', 'retrievers')

# Đảm bảo thư mục tồn tại
os.makedirs(RETRIEVER_FOLDER, exist_ok=True)

def save_retriever(course_id, lesson_number, retriever, summaries):
    """
    Lưu retriever và summaries vào file
    
    Args:
        course_id: ID của khóa học
        lesson_number: Số buổi học
        retriever: Đối tượng retriever
        summaries: Tóm tắt nội dung
    """
    try:
        retriever_key = f"{course_id}_L{lesson_number}"
        
        # Tạo thư mục cho retriever này
        retriever_dir = os.path.join(RETRIEVER_FOLDER, retriever_key)
        os.makedirs(retriever_dir, exist_ok=True)
        
        # Tạo thư mục chroma_db
        chroma_dir = os.path.join(retriever_dir, "chroma_db")
        os.makedirs(chroma_dir, exist_ok=True)
        
        # Lưu vectorstore - cách tiếp cận khác
        # Lấy embedding function từ vectorstore hiện tại
        embedding_function = retriever.vectorstore._embedding_function
        
        # Tạo một vectorstore mới với persist_directory được chỉ định
        from langchain.vectorstores import Chroma
        
        # Lấy tất cả các documents từ vectorstore hiện tại
        all_docs = []
        
        # Lấy documents từ vectorstore similarity search
        try:
            # Tìm kiếm với query rộng để lấy tất cả documents
            docs = retriever.vectorstore.similarity_search("", k=1000)
            all_docs.extend(docs)
        except:
            pass
            
        # Tạo vectorstore mới
        new_vectorstore = Chroma(
            collection_name="multi_modal_rag",
            embedding_function=embedding_function,
            persist_directory=chroma_dir
        )
        
        # Thêm documents nếu có
        if all_docs:
            new_vectorstore.add_documents(all_docs)
            new_vectorstore.persist()
            
        # Lưu docstore với cách an toàn hơn
        docstore_data = {}
        if hasattr(retriever.docstore, '_dict'):
            docstore_data = retriever.docstore._dict.copy()
        elif hasattr(retriever.docstore, 'store'):
            docstore_data = retriever.docstore.store.copy()
        
        docstore_path = os.path.join(retriever_dir, "docstore.pkl")
        with open(docstore_path, 'wb') as f:
            pickle.dump(docstore_data, f, pickle.HIGHEST_PROTOCOL)
        
        # Lưu summaries
        summaries_path = os.path.join(retriever_dir, "summaries.pkl")
        with open(summaries_path, 'wb') as f:
            pickle.dump(summaries, f, pickle.HIGHEST_PROTOCOL)
        
        # Lưu thông tin cấu hình
        config_path = os.path.join(retriever_dir, "config.pkl")
        config = {
            "id_key": retriever.id_key,
            "collection_name": "multi_modal_rag",
            "embedding_type": str(type(embedding_function))
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
            
        logger.info(f"Đã lưu retriever {retriever_key} thành công")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu retriever {retriever_key}: {str(e)}")
        return False

def load_retriever(course_id, lesson_number):
    try:
        retriever_key = f"{course_id}_L{lesson_number}"
        retriever_dir = os.path.join(RETRIEVER_FOLDER, retriever_key)
        
        if not os.path.exists(retriever_dir):
            return None, None
        
        # Tải vectorstore
        chroma_dir = os.path.join(retriever_dir, "chroma_db")
        embedding_function = OpenAIEmbeddings()
        
        vectorstore = Chroma(
            collection_name="multi_modal_rag",
            embedding_function=embedding_function,
            persist_directory=chroma_dir
        )
        
        # Tải docstore
        docstore_path = os.path.join(retriever_dir, "docstore.pkl")
        with open(docstore_path, 'rb') as f:
            docstore_data = pickle.load(f)
        
        # Tạo lại docstore
        from langchain.storage import InMemoryStore
        docstore = InMemoryStore()
        
        # Khôi phục dữ liệu docstore
        if isinstance(docstore_data, dict):
            for key, value in docstore_data.items():
                docstore.mset([(key, value)])
        
        # Tạo retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id",
        )
        
        # Tải summaries
        summaries_path = os.path.join(retriever_dir, "summaries.pkl")
        if os.path.exists(summaries_path):
            with open(summaries_path, 'rb') as f:
                summaries = pickle.load(f)
        else:
            summaries = []
            
        return retriever, summaries
        
    except Exception as e:
        logger.error(f"Lỗi khi tải retriever: {str(e)}")
        return None, None

def delete_retriever(course_id, lesson_number=None):
    """
    Xóa retriever của một buổi học hoặc tất cả các buổi của một khóa học
    
    Args:
        course_id: ID của khóa học
        lesson_number: Số buổi học (None để xóa tất cả các buổi)
    """
    try:
        if lesson_number is not None:
            # Xóa retriever của một buổi cụ thể
            retriever_key = f"{course_id}_L{lesson_number}"
            retriever_dir = os.path.join(RETRIEVER_FOLDER, retriever_key)
            
            if os.path.exists(retriever_dir):
                import shutil
                shutil.rmtree(retriever_dir)
                logger.info(f"Đã xóa retriever {retriever_key}")
        else:
            # Xóa tất cả retrievers của khóa học
            for item in os.listdir(RETRIEVER_FOLDER):
                if item.startswith(f"{course_id}_L"):
                    retriever_dir = os.path.join(RETRIEVER_FOLDER, item)
                    if os.path.isdir(retriever_dir):
                        import shutil
                        shutil.rmtree(retriever_dir)
                        logger.info(f"Đã xóa retriever {item}")
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi xóa retriever: {str(e)}")
        return False

def list_retrievers():
    """
    Liệt kê tất cả các retrievers đã lưu
    
    Returns:
        list: Danh sách các khóa retriever
    """
    try:
        retrievers = []
        for item in os.listdir(RETRIEVER_FOLDER):
            retriever_dir = os.path.join(RETRIEVER_FOLDER, item)
            if os.path.isdir(retriever_dir):
                # Kiểm tra xem thư mục có chứa các file cần thiết không
                if os.path.exists(os.path.join(retriever_dir, "chroma_db")) and \
                   os.path.exists(os.path.join(retriever_dir, "docstore.pkl")):
                    retrievers.append(item)
        
        return retrievers
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê retrievers: {str(e)}")
        return []
