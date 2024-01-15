

function [Imlab,M_neig] = Reg_BBA(M_data,d)

it=0;
change_ok = 1;
Imlab = Decision(M_data);

dim=log2(size(M_data,3));

figure;
subplot(1,2,1),imshow(Imlab,[]), title('Initialisation');

while (it<10 && change_ok)
    
    M_neig= zeros(size(M_data));
    
    for i=1:size(M_data,1)
        for j=1:size(M_data,2)
            
            n_neig=zeros(1,dim);
            nn=0;
            
            for ii=max(i-d,1):min(i+d,size(M_data,1))
                for jj=max(j-d,1):min(j+d,size(M_data,2))
                    
                    n_neig(Imlab(ii,jj))=n_neig(Imlab(ii,jj))+1;
                    nn=nn+1;
                    
                end
            end
            
            n_neig(Imlab(i,j)) = n_neig(Imlab(i,j))-1;
            nn = nn-1;
            
            [n_neig_sort,ind]=sort(n_neig);
            indH=0;
            
            for k=dim:-1:2
                x=(n_neig_sort(k)-n_neig_sort(k-1))/nn;
                indH=indH+2^(ind(k)-1);
                M_neig(i,j,indH)=x*(dim+1-k);
            end
            
            M_neig(i,j,size(M_data,3)-1)=n_neig_sort(1)/nn*dim;
            
        end
    end
    
    Im_m = Regle_Dempster(M_data,M_neig);
    Imlab2=Decision(Im_m);
    
    if (Imlab==Imlab2)
        change_ok = 0;
    else
        Imlab=Imlab2;
    end
    %     figure;imshow(Imlab/2)
    %     figure;imshow(M_neig(:,:,1))
    %     figure;imshow(M_neig(:,:,2))
    %     figure;imshow(M_neig(:,:,3))
    %     figure;imshow(M_neig(:,:,4))
    %     figure;imshow(Im_m(:,:,size(M_data,3)))
    
    it=it+1;
    
    subplot(1,2,2),imshow(Imlab,[]), title(['Iteration ' num2str(it)]);
    pause(0.5);
    
end

