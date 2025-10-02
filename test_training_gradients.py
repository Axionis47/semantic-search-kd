"""Quick test to verify training gradients work."""

import torch
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.kd.losses import CombinedKDLoss

def test_gradients():
    """Test that gradients flow through the student model."""
    print("=" * 80)
    print("Testing Training Gradients")
    print("=" * 80)
    
    # Initialize models
    print("\n1. Loading models...")
    student = StudentModel("intfloat/e5-small-v2", device="cpu")
    teacher = TeacherModel("BAAI/bge-reranker-large", device="cpu")
    
    # Set student to training mode
    student.model.train()
    print("✓ Models loaded")
    
    # Create simple test data
    print("\n2. Creating test data...")
    query = "What is machine learning?"
    docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "The sky is blue.",
    ]
    print(f"✓ Query: {query}")
    print(f"✓ Docs: {len(docs)}")
    
    # Encode with gradients
    print("\n3. Encoding with gradients...")
    query_emb = student.encode_with_gradients([query], normalize=True)
    doc_embs = student.encode_with_gradients(docs, normalize=True)
    print(f"✓ Query embedding shape: {query_emb.shape}")
    print(f"✓ Doc embeddings shape: {doc_embs.shape}")
    print(f"✓ Query embedding requires_grad: {query_emb.requires_grad}")
    print(f"✓ Doc embeddings requires_grad: {doc_embs.requires_grad}")
    
    # Compute student scores
    print("\n4. Computing student scores...")
    student_scores = torch.matmul(query_emb, doc_embs.T)[0]
    print(f"✓ Student scores: {student_scores}")
    print(f"✓ Student scores requires_grad: {student_scores.requires_grad}")
    
    # Get teacher scores
    print("\n5. Getting teacher scores...")
    pairs = [[query, doc] for doc in docs]
    teacher_scores = teacher.score(pairs)
    teacher_scores_tensor = torch.tensor(teacher_scores, dtype=torch.float32)
    print(f"✓ Teacher scores: {teacher_scores_tensor}")
    
    # Compute loss
    print("\n6. Computing loss...")
    loss_fn = CombinedKDLoss(
        margin_mse_weight=0.6,
        listwise_kd_weight=0.2,
        contrastive_weight=0.2,
        temperature_start=4.0,
        temperature_end=2.0,
    )
    
    loss_dict = loss_fn(
        student_scores.unsqueeze(0),
        teacher_scores_tensor.unsqueeze(0)
    )
    
    total_loss = loss_dict["loss"]
    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ Loss requires_grad: {total_loss.requires_grad}")
    print(f"✓ Loss has grad_fn: {total_loss.grad_fn is not None}")
    
    # Test backward pass
    print("\n7. Testing backward pass...")
    try:
        total_loss.backward()
        print("✓ Backward pass successful!")
        
        # Check if gradients were computed
        has_grads = False
        for name, param in student.model.named_parameters():
            if param.grad is not None:
                has_grads = True
                print(f"✓ Gradient computed for: {name[:50]}...")
                break
        
        if has_grads:
            print("\n" + "=" * 80)
            print("SUCCESS! Training gradients work correctly!")
            print("=" * 80)
            return True
        else:
            print("\n" + "=" * 80)
            print("WARNING: No gradients found in model parameters")
            print("=" * 80)
            return False
            
    except Exception as e:
        print(f"\n✗ Backward pass failed: {e}")
        print("=" * 80)
        print("FAILED! Gradients do not flow correctly")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = test_gradients()
    exit(0 if success else 1)

